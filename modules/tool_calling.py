import gradio as gr
import json
import re
import uuid

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
                #tool_call = message[indices[i][0]:indices[i][1]]
                # At least in Llama 3, putting content in the same message as the tool call actually throws out the content...
                #messages.append({"role": "assistant", "content": assistant_message, "tool_calls": [json_object]})
                if assistant_message != "":
                    messages.append({"role": "assistant", "content": assistant_message})
                messages.append({"role": "assistant", "content": "", "tool_calls": [json_object]})
            # Tool response
            elif 'tool_call_id' in json_object:
                #tool_response = message[indices[i][0]:indices[i][1]]
                #messages.append({"role": "tool", "content": tool_response})
                messages.append(json_object)
            else:
                print("Invalid JSON object found in response")
        assistant_message = message[indices[-1][1]:]
        #if assistant_message != "":
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
                        #print(json_str)
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
    # Only needs to run when the tool is activated (when you check the box in the UI to enable it in the current session)
    # This has a slight downside of potentially overwriting locals though...
    if 'name' not in tool:
        print("Tool needs a name!")
        return False
    if 'action' in tool:
        #print("Action found")
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
                #print("Function:", defined_function)
                if defined_function not in tool_defined_params and (defined_function in globals().keys() or defined_function in locals().keys()):
                    print(f"Security warning: Tool action function '{defined_function}' attempts to override existing global!")
                    valid = False
                    break
                defined_params.append(defined_function)
            for defined_variable in re.findall(r"""^([^\s]+)\s*=""", action_code, flags=re.M):
                #print("Variable:", defined_variable)
                if defined_variable not in tool_defined_params and (defined_variable in globals().keys() or defined_variable in locals().keys()):
                    print(f"Security warning: Tool action variable '{defined_variable}' attempts to override existing global!")
                    valid = False
                    break
                defined_params.append(defined_variable)
            # Check for eval/exec to warn the user (not really necessary, just to be safe, doesn't prevent it from being used). Code interpreter tools will use this, for example.
            tool_defined_params = tool_defined_params | set(defined_params)
            print(tool_defined_params)
            if 'exec' in action_code or 'eval' in action_code:
                gr.Warning("This tool may allow for arbitrary code execution, be careful!")
            if valid:
                print(f"Defining tool: {tool['name']}")
                # TODO: There's no reason to redefine the tool each time it needs to run, right? This exec call only needs to be done when the tool is activated.
                exec(action_code, globals()) # DANGEROUS! You could overwrite existing functions this way... And obviously this is arbitrary code execution. So please make sure the code you're running is safe.
                # Assuming the function matching the name of the tool was defined?
                action_function = globals()[tool['name']]
                #print(action_function, type(action_function))
                if action_function is not None and str(type(action_function)) == "<class 'function'>":
                    print(f"Tool {tool['name']} action defined.")
                    return True
        except Exception as e:
            print(f"Exception while defining tool {tool['name']}")
            print(e)
    else:
        print("Tool {tool['name'] has no action defined.")
    return False


def validate_tool_input_parameters(json_object, tool):
    valid = True
    if 'parameters' in json_object and json_object['parameters'] is not None:
        if type(json_object['parameters']) == str:
            try:
                json_object['parameters'] = json.loads(json_object['parameters'])
            except Exception:
                print("Invalid parameters format")
                return False
        input_params = set(json_object['parameters'].keys())
        if 'required' in tool['parameters']:
            req_params = set(tool['parameters']['required'])
            if not req_params.issubset(input_params):
                print(f"Tool call missing required parameters: {req_params - input_params}")
                return False
        if 'properties' in tool['parameters']:
            for property, metadata in tool['parameters']['properties'].items():
                if property in input_params:
                    # Check type
                    input_value = json_object['parameters'][property]
                    #print(input_value, type(input_value))
                    if 'type' in metadata:
                        if metadata['type'] == 'number':
                            try:
                                json_object['parameters'][property] = float(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type number")
                                valid = False
                                break
                        if metadata['type'] == 'string':
                            try:
                                json_object['parameters'][property] = str(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type string")
                                valid = False
                                break
                        if metadata['type'] == 'bool':
                            try:
                                json_object['parameters'][property] = bool(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type bool")
                                valid = False
                                break
                        if metadata['type'] == 'list':
                            try:
                                json_object['parameters'][property] = list(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type list")
                                valid = False
                                break
                        if metadata['type'] == 'object':
                            try:
                                json_object['parameters'][property] = dict(input_value)
                            except Exception:
                                print(f"Invalid value: {input_value} not of type object")
                                valid = False
                                break
                        # TODO: Validation of object fields...
        # Function call with no arguments
        if len(input_params) == 0:
            json_object['parameters'] = None
    else:
        # Function takes no parameters
        json_object['parameters'] = None
    return valid


def process_tool_calls(response, tools):
    tool_calls = []
    #results = []
    if tools is None or len(tools) == 0:
        return response, tool_calls #, results
    
    # Parse response for JSON output?
    # There may already be a library for this that I'm not aware of
    json_objects, indices = extract_json_from_response(response)
    modified_message = ""
    
    if len(json_objects) > 0:
        #modified_message = response[:indices[0][0]] + "\n\n" if len(response[:indices[0][0]]) > 0 else ""

        #print('Found JSON in response')
        #print(json_objects)
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
                    # Tool name match
                    #print(f"Tool {tool['name']} name match")
                    # Validate parameters
                    # Side effects - modifies the JSON object with type conversions
                    valid = validate_tool_input_parameters(json_object, tool)
                    if not valid:
                        continue
                    # Valid arguments? Check the "additionalProperties" field?
                    #print("Validation passed!")
                    #print("Parameters:", json_object['parameters'])
                    
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
                        tool_calls.append({
                            "id": tool_call_id,
                            "type": tool['type'],
                            tool['type']: {
                                "name": tool['name'],
                                #"arguments": json.dumps(json_object['parameters']) if json_object['parameters'] is not None else None
                                "arguments": json_object['parameters'] if json_object['parameters'] is not None else None
                            }
                        })
                        '''
                        try:
                            
                            action_function = globals()[tool['name']]
                            result = None
                            # TODO: Sandbox this somehow
                            if json_object['parameters'] is not None:
                                result = action_function(json_object['parameters'])
                            else:
                                result = action_function()
                            
                            results.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": str(result)
                            })
                        except Exception as e:
                            print(f"Exception while running action for tool {tool['name']} with inputs {json_object['parameters']}")
                            print(e)
                            tool_calls.append({
                                "id": tool_call_id,
                                "type": tool['type'],
                                tool['type']: {
                                    "name": tool['name'],
                                    #"arguments": json.dumps(json_object['parameters']) if json_object['parameters'] is not None else None
                                    "arguments": json_object['parameters'] if json_object['parameters'] is not None else None
                                }
                            })
                            results.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": "Error" # Should the error message be included?
                            })
                        '''
            if not valid_tool_found:
                # Add back in the extracted JSON object
                modified_message += response[indices[i][0]:indices[i][1]]

        # Add the last part of the response
        modified_message += response[indices[-1][1]:]

    print("Tool Calls:")
    for tool_call in tool_calls:
        print(json.dumps(tool_call))
    
    # Only return tool calls, wait for results
    '''
    print("Results:")
    for result in results:
        print(json.dumps(result))
    '''
    return modified_message, tool_calls #, results


def extract_tool_calls_to_be_executed(message, tools):
    tool_call_ids = set()
    tool_response_ids = set()
    json_objects, indices = extract_json_from_response(message)
    if len(json_objects) > 0:
        for i, json_object in enumerate(json_objects):
            if 'id' in json_object:
                # Tool Call
                tool_call_ids.add(json_object['id'])
            elif 'tool_call_id' in json_object:
                # Tool Response
                tool_response_ids.add(json_object['tool_call_id'])
        tool_call_ids_to_execute = tool_call_ids - tool_response_ids
        tool_calls_to_execute = []
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
            # TODO: Sandbox this somehow
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

tools = [
	{
		"action": "# Assumed input is {\"latitude\": 90.0, \"longitude\": 30.0\"}\ntest_global = 5\ndef get_weather(parameters):\n  test_local = 4\n  print(parameters['latitude'], \n  parameters['longitude'])\n  return 0.0",
		"description": "Get current temperature for provided coordinates in celsius.",
		"name": "get_weather",
		"parameters": {
			"additionalProperties": False,
			"properties": {
				"latitude": {
					"type": "number"
				},
				"longitude": {
					"type": "number"
				}
			},
			"required": [
				"latitude",
				"longitude"
			],
			"type": "object"
		},
		"type": "function"
	},
    
    {
        "action": "import random\ndef random_number():\n  return random.random()",
        "description": "Get a random number between 0 and 1",
        "name": "random_number",
        "parameters": {},
        "type": "function"
    }
]

# Testing

'''
response = """
{"name": "get_weather",
"parameters": {"latitude": "47.6067", "longitude": "122.3322"},
"type": "function"}}
"""

process_tool_calls(response, tools)
'''