import gradio as gr
import json
import re
import uuid
import traceback

# Keep track of the functions and variables defined by tool actions as part of security check preventing overwriting of globals
tool_defined_params = set()

def split_message_by_tool_calls(message):
    messages = []
    json_objects, indices = extract_json_from_response(message)
    # Preprocess to remove the JSON objects that aren't tool calls
    not_tool_related = []
    for i, json_object in enumerate(json_objects):
        if 'id' not in json_object and 'tool_call_id' not in json_object:
            not_tool_related.append(i)
    for i in not_tool_related[::-1]:
        del json_objects[i]
    if len(json_objects) > 0:
        for i, json_object in enumerate(json_objects):
            # Tool call
            if 'id' in json_object:
                start_index = indices[i-1][1] if i > 0 else 0
                end_index = indices[i][0]
                assistant_message = message[start_index:end_index] if len(message[start_index:end_index].strip()) > 0 else ""
                # At least in Llama 3, putting content in the same message as the tool call actually throws out the content...
                #messages.append({"role": "assistant", "content": assistant_message, "tool_calls": [json_object]})
                if assistant_message != "":
                    messages.append({"role": "assistant", "content": assistant_message})
                messages.append({"role": "assistant", "content": "", "tool_calls": [json_object]})
            # Tool response
            elif 'tool_call_id' in json_object:
                messages.append(json_object)
            else:
                print("Invalid JSON object found in response")
        assistant_message = message[indices[-1][1]:]
        #if assistant_message != "": # I think this is necessary even if it's an empty message to ensure the next message role isn't tool/ipython
        messages.append({"role": "assistant", "content": assistant_message})
    elif message != "":
        messages.append({"role": "assistant", "content": message})
    return messages


# Don't make the tool action code visible to the LLM
def get_visible_tools(state):
    if 'tools' not in state or state['tools'] is None:
        return None
    if len(state['tools']) == 0:
        return None
    visible_tools = []
    for tool in state['tools']:
        visible_tool = {k: v for (k, v) in tool.items() if k != 'action'}
        visible_tools.append(visible_tool)
    return visible_tools


def extract_json_from_response(response):
    json_objects = []
    indices = []
    stack = []
    start_index = 0
    for i, char in enumerate(response):
        if char == '{':
            if not stack:
                start_index = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    json_str = response[start_index:i+1]
                    indices.append((start_index, i+1))
                    try:
                        json_obj = json.loads(json_str)
                        json_objects.append(json_obj)
                    except json.decoder.JSONDecodeError as e:
                        print('Failed to parse JSON in response')
    return json_objects, indices


# Not sure what the usual format for this ID is
def generate_tool_call_id():
    return uuid.uuid4().hex[:8]


def define_tool_action(tool):
    global tool_defined_params
    # TODO: There seems to be a bug where selecting a different tool will have the previous tool still visible to the model for one message.
    # Only needs to run when the tool is activated (when you check the box in the UI to enable it in the current session)
    # This has a slight downside of potentially overwriting locals though...
    if 'name' not in tool:
        print("Tool needs a name!")
        return False
    if 'action' in tool:
        try:
            action_code = tool['action']
            #print("Tool action code:")
            #print(action_code)
            # Basic validation of action code
            # Check for existence of function definitions and ensure they don't overwrite globals? This isn't perfect but should help at least a little.
            # This also doesn't prevent any kind of backdoors or other security issues in custom code, so just make sure you check whether the code you're running with tools is reasonable...
            valid = True
            defined_params = []
            for defined_function in re.findall(r"""^def\s+([^\(\s]+)""", action_code, flags=re.M):
                if defined_function not in tool_defined_params and (defined_function in globals().keys() or defined_function in locals().keys()):
                    print(f"Security warning: Tool action function '{defined_function}' attempts to override existing global!")
                    valid = False
                    break
                defined_params.append(defined_function)
            for defined_variable in re.findall(r"""^([^\s]+)\s*=""", action_code, flags=re.M):
                if defined_variable not in tool_defined_params and (defined_variable in globals().keys() or defined_variable in locals().keys()):
                    print(f"Security warning: Tool action variable '{defined_variable}' attempts to override existing global!")
                    valid = False
                    break
                defined_params.append(defined_variable)
            tool_defined_params = tool_defined_params | set(defined_params)
            #print(tool_defined_params)
            # Check for eval/exec to warn the user (not really necessary, just to be safe, doesn't prevent it from being used). Code interpreter tools will use this, for example.
            # Obviously there are a lot of ways to hide this function and searching for the text directly isn't guaranteed to find it, but I'm going to assume the user understands whatever tool action code they have entered.
            if 'exec' in action_code or 'eval' in action_code:
                gr.Warning("This tool may allow for arbitrary code execution, be careful!")
            if valid:
                print(f"Defining tool: {tool['name']}")
                # TODO: Sandbox this somehow, as this can be extremely dangerous. But for now, it depends on the user to ensure the code is safe.
                exec(action_code, globals()) # DANGEROUS! You could overwrite existing functions this way... And obviously this is arbitrary code execution. So please make sure the code you're running is safe.
                # Assuming the function matching the name of the tool was defined?
                action_function = globals()[tool['name']]
                if action_function is not None and str(type(action_function)) == "<class 'function'>":
                    print(f"Tool {tool['name']} action defined.")
                    return True
        except Exception as e:
            print(f"Exception while defining tool {tool['name']}")
            print(traceback.print_exc())
    else:
        print("Tool {tool['name'] has no action defined.")
    return False


def validate_tool_input_parameters(json_object, tool):
    valid = True
    # Determine if 'arguments' or 'parameters' is used
    tool_params = None
    if 'parameters' in json_object:
        tool_params = json_object['parameters']
    elif 'arguments' in json_object:
        tool_params = json_object['arguments']

    if tool_params is not None:
        if type(tool_params) == str:
            try:
                tool_params = json.loads(tool_params)
            except Exception:
                print("Invalid parameters format")
                return False
        input_params = set(tool_params.keys())
        if 'required' in tool['parameters']:
            req_params = set(tool['parameters']['required'])
            if not req_params.issubset(input_params):
                print(f"Tool call missing required parameters: {req_params - input_params}")
                return False
        if 'properties' in tool['parameters']:
            for property, metadata in tool['parameters']['properties'].items():
                if property in input_params:
                    # Check type
                    input_value = tool_params[property]
                    #print(input_value, type(input_value))
                    if 'type' in metadata:
                        if metadata['type'] == 'number':
                            try:
                                tool_params[property] = float(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type number")
                                valid = False
                                break
                        if metadata['type'] == 'string':
                            try:
                                tool_params[property] = str(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type string")
                                valid = False
                                break
                        if metadata['type'] == 'bool':
                            try:
                                tool_params[property] = bool(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type bool")
                                valid = False
                                break
                        if metadata['type'] == 'list':
                            try:
                                tool_params[property] = list(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type list")
                                valid = False
                                break
                        if metadata['type'] == 'object':
                            try:
                                tool_params[property] = dict(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type object")
                                valid = False
                                break
                        # TODO: Validation of object fields (with nested properties)...
                        # TODO: Check the "additionalProperties" field?
        # Function call with no arguments
        if len(input_params) == 0:
            tool_params = None
    else:
        # Function takes no parameters
        tool_params = None # Redundant
    return valid


def process_tool_calls(response, tools):
    tool_calls = []
    if tools is None or len(tools) == 0:
        return response, tool_calls
    
    # Parse response for JSON output?
    # There may already be a library for this that I'm not aware of
    json_objects, indices = extract_json_from_response(response)
    modified_message = ""
    
    if len(json_objects) > 0:
        for i, json_object in enumerate(json_objects):
            start_index = indices[i-1][1] if i > 0 else 0
            end_index = indices[i][0]
            modified_message += response[start_index:end_index] if len(response[start_index:end_index].strip()) > 0 else ""
            # Check whether they fit the required format of any of the tools
            if 'name' not in json_object:
                print('JSON missing required fields for tool call')
                modified_message += response[indices[i][0]:indices[i][1]]
                continue
            valid_tool_found = False
            for tool in tools:
                if 'name' not in tool or 'parameters' not in tool:
                    print('Invalid tool definition')
                    continue
                if json_object['name'] == tool['name']:
                    # Validate parameters
                    # Side effects - modifies the JSON object with type conversions
                    valid = validate_tool_input_parameters(json_object, tool)
                    if not valid:
                        continue
                    
                    # This doesn't take into account whether the function code has been updated
                    # Only whether it was initially defined
                    # Really we should just ensure that the definition runs when enabling the tool
                    tool_action_defined = False
                    if not tool['name'] in globals():
                        tool_action_defined = define_tool_action(tool)
                    elif str(type(globals()[tool['name']])) == "<class 'function'>":
                        tool_action_defined = True
                    if tool_action_defined:
                        tool_call_id = f"call_{tool['name']}_{generate_tool_call_id()}"
                        valid_tool_found = True
                        tool_params = None
                        if 'parameters' in json_object:
                            tool_params = json_object['parameters']
                        elif 'arguments' in json_object:
                            tool_params = json_object['arguments']

                        tool_calls.append({
                            "id": tool_call_id,
                            "type": tool['type'],
                            tool['type']: {
                                "name": tool['name'],
                                "arguments": tool_params,
                            }
                        })
            if not valid_tool_found:
                # Add back in the extracted JSON object
                modified_message += response[indices[i][0]:indices[i][1]]

        # Add the last part of the response
        modified_message += response[indices[-1][1]:]

    print("Tool Calls:")
    for tool_call in tool_calls:
        print(json.dumps(tool_call))
    
    # Only return tool calls, wait for results separately, as it depends whether we have the setting active for requiring the user to confirm
    # The modified message here removes the raw tool call JSON in order to replace it with the one including the generated ID
    return modified_message, tool_calls


def extract_tool_calls_to_be_executed(message, tools):
    tool_call_ids = set()
    tool_response_ids = set()
    json_objects, indices = extract_json_from_response(message)
    tool_calls_to_execute = []
    if len(json_objects) > 0:
        for i, json_object in enumerate(json_objects):
            if 'id' in json_object:
                # Tool Call
                tool_call_ids.add(json_object['id'])
            elif 'tool_call_id' in json_object:
                # Tool Response
                tool_response_ids.add(json_object['tool_call_id'])
        tool_call_ids_to_execute = tool_call_ids - tool_response_ids
        for i, json_object in enumerate(json_objects):
            if 'id' in json_object and json_object['id'] in tool_call_ids_to_execute:
                tool_calls_to_execute.append(json_object)
    return tool_calls_to_execute


def execute_tool_call(tool_call):
    try:
        tool_call_id = tool_call['id']
        tool_call_type = tool_call['type']
        if tool_call_type in tool_call:
            tool = tool_call[tool_call_type]               
            action_function = globals()[tool['name']]
            result = None
            # TODO: Sandbox this somehow, as this can be extremely dangerous, especially if the user is executing LLM-generated code without reviewing it.
            # But for now, it depends on the user to ensure the code is safe. And some tools can be used completely safely, it's mostly things like code interpreter that can be problematic.
            # TODO: Code interpreter outputs to the console right now, needs to have the input and output displayed in the UI somehow.
            if tool['arguments'] is not None:
                result = action_function(tool['arguments'])
            else:
                result = action_function()
            print(result)
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(result)
            }
        raise Exception("Invalid tool call type")
    except Exception as e:
        print(f"Exception while running action for tool {tool['name']} with inputs {tool['arguments']}")
        print(e)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": "Error" # Should the error message be included?
        }

