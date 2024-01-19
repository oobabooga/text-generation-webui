from extensions.openai.errors import InvalidRequestError
from extensions.openai.utils import debug_msg
from extensions.openai.typing import FunctionCallRequest, FunctionCallResponse, FunctionNameArg
from typing import List

"""
This is a class to hold the context of a function call

Essentially, the function calling is simply user adding functions with json format into url request body, and server process these functions into a part of system prompt. 
Either with 1 shot prompting or some sft function calling ability from the base llm, a llm model is able spit out function api name and arguments in json format. Finally, the server post process the llm ouput into url reponse body.

Done:
- Support both function calling(deprecated, but some project like memgpt still use it!) and tool calling api for openai chat completion api (shold not support stream though, but I am not sure)
- Implement function calling context that parse user functions and function calls into sysmte prompt,
- Handle function role as part of historical user input
- Post process function calling finish reason
- Very basic run in the above openai cook book example, and it worked (though the response is not as accurate as chatgpt3.5 or above)!

Caveats:
- Only support generating single function calling in a single response so far. Parsing multi function call json str is simply not implemented yet. That is because I have not seen any example in the cookbook where multi function calling in a single response, though the openai response format implies there could be multi function calling (as it uses a list to store every function calling response)
- I personally use a xml wrapper for <functioncall> and for function role, I used <functionresponse> as indicators. Those indicators I also used in training sft model for function calling. THIS IS NOT UNIVERSAL, and base models that have not exposed with glaive-function-calling-v2 might not follow it even with 1/N shot prompting
"""
class FunctionCallContext():
    def __init__(self, body):
        
        def init_from_body(body):
            self.functions = body.get(self.FUNCTIONS, [])
            self.function_call = body.get(self.FUNCTION_CALL, 'none')
            
            if self.functions is None:
                self.functions = []
            if self.function_call is None:
                # if functions is empty, function_call defaults to none
                # https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
                if self.functions == []:
                    self.function_call = 'none'
                # otherwise, if functions are provided, default function_call to auto
                else:
                    self.function_call = 'auto'
                
            # Validation
            if self.functions != [] and self.function_call == 'none':
                # If user specify function_call to none even when functions are provided, we ignore all functions
                # https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
                self.functions = []
            if self.functions == [] and (self.function_call != 'none' and self.function_call != 'auto'):
                raise InvalidRequestError(message=f"function_call {self.function_call} is provided but functions are none", param=self.FUNCTIONS)
        
        # Legacy function call api
        self.use_legacy = True
        if self.use_legacy:
            self.ROLE = 'function'
            self.FINISH_REASON = self.RESPOSE ='function_call'
            self.FUNCTIONS = 'functions'
            self.FUNCTION_CALL = 'function_call'
            init_from_body(body)

        # See if lecagy is used, otherwise try new format
        if self.functions != [] and self.function_call != 'none':
            self.use_legacy = True
        else:
            self.use_legacy = False
            
        # Try use new format https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
        if not self.use_legacy:
            self.ROLE = 'tool'
            self.FINISH_REASON = self.RESPOSE ='tool_calls'
            self.FUNCTIONS = 'tools'
            self.FUNCTION_CALL = 'tool_choice'
            init_from_body(body)
            
        # Handle function requests
        if self.functions != []:
            self.expect_finish_with_function_call = True
            self.FUNCTION_PROMPT = 'You are given access to the following functions, use them if required -'
        
            for func in self.functions:
                if not FunctionCallRequest.model_validate(func):
                    raise InvalidRequestError(message=f"function {func} is not a valid format", param='functions')
        
            if self.function_call != 'auto':
                # check if function_call is correct format
                if not FunctionCallRequest.model_validate(self.function_call):
                    raise InvalidRequestError(message=f"function_call {self.function_call} is not valid format", param='function_call')
                
                # check if function_call is in functions, and only add selected function to prompt
                found_function_call = False
                for func in self.functions:
                    if FunctionCallRequest.model_validate(func).function.name == FunctionCallRequest.model_validate(self.function_call).function.name:
                        found_function_call = True
                        self.FUNCTION_PROMPT += f'\n{func["function"]}\n'
                        break
                if not found_function_call:
                    raise InvalidRequestError(message=f"function_call {self.function_call} is not in functions", param='function_call')
                
                # must call function
                self.FUNCTION_PROMPT += f'\nYOU MUST CALL THIS FUNCTION IN YOUR REPLY: {self.function_call["function"]}.'
            else:            
                for func in self.functions:
                    self.FUNCTION_PROMPT += f'\n{func["function"]}\n'
                    
            # give 1-shot prompt for function call reply format
            self.FUNCTION_PROMPT += """If you find it necessary to call function, you must reply in the format only when necessary: <functioncall> json_str </functioncall>, e.g <functioncall> {\"name\": \"calculate_loan_payment\", \"arguments\": '{\"principal\": 50000, \"interest_rate\": 5, \"loan_term\": 10}'} </functioncall>."""
            self.FUNCTION_PROMPT += """Here is a sample conversation between a helpful assistant that call a function for user requests that are within range of provided functions, and reject invalid user request that provided functions cannot do:
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{\n \"name\": \"calculate_median\",\n \"description\": \"Calculate the median of a list of numbers\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"numbers\": {\n \"type\": \"array\",\n \"items\": {\n \"type\": \"number\"\n },\n \"description\": \"A list of numbers\"\n }\n },\n \"required\": [\n \"numbers\"\n ]\n }\n}
USER: Hi, I have a list of numbers and I need to find the median. Can you help me with that?
ASSISTANT: Of course, I can help you with that. Please provide me with the list of numbers.
USER: The numbers are 5, 2, 9, 1, 7, 4, 6, 3, 8.
ASSISTANT: <functioncall> {"name": "calculate_median", "arguments": '{"numbers": [5, 2, 9, 1, 7, 4, 6, 3, 8]}'} </functioncall>
USER: <functionresponse> {"median": 5} </functionresponse>
ASSISTANT: The median of the list of numbers is 5.
USER: That's great! Can you also help me book a flight to Bei Jing next week?
ASSISTANT: I'm sorry, I don't have the capability to book flights or perform external tasks. My current task is to calculate median number with a list of numbers.
End of the example convseration.
"""
        else:
            self.expect_finish_with_function_call = False
            self.FUNCTION_PROMPT = ''
            
    def process_role_msg(self, content)->str:
        return f'Previous function call has responded with: <functionresponse> {content} </functionresponse>.'
    
    def process_finish_msg(self, content)-> (List[dict], bool):
        finish_responses = []
        exception_occurred = False
        try:
            if self.functions != []:
                import re
                import ast
                import json
                # Define the pattern to match the JSON string within the functioncall tags
                pattern = r'<functioncall>(.*?)</functioncall>'

                debug_msg(f"process_finish_msg Try match '{pattern}' from llm reply '{content}'")
                # Use re.findall to find all matched patterns
                matches = re.findall(pattern, content, re.DOTALL)
                if len(matches) == 0:
                    debug_msg("process_finish_msg No match found.")
                    
                for match in matches:
                    if match is None or not isinstance(match, str):
                        debug_msg(f"process_finish_msg match is None or not str {match}")
                        continue

                    json_str = match.strip()
                    debug_msg(f"process_finish_msg Found matching json_str {json_str}")

                    json_dict = None
                    try:
                        """
                        https://www.datasciencebyexample.com/2023/03/16/what-to-do-when-single-quotes-in-json-string/
                        llm function call response actually is hard to parse using json.load or pydantic.parse_raw due to the value of `argument` key in nasty single quote format
                        e.g {\"name\": \"calculate_loan_payment\", \"arguments\": '{\"principal\": 50000, \"interest_rate\": 5, \"loan_term\": 10}'}
                        so we have to use ast.literal_eval
                        """
                        json_dict = ast.literal_eval(json_str)
                    except SyntaxError as e:
                        debug_msg(f"process_finish_msg Error parsing json_str {json_str} with error {e}")
                    
                    if json_dict is None or not isinstance(json_dict, dict):
                        try:
                            debug_msg(f"process_finish_msg json_dict is None or not dict, will try method 2 {json_dict}")
                            """
                            Try to replace single quote with triple quote
                            Edge sample below could be resolved by this method
                            {"name": "ask_database", "arguments": '{"query": "SELECT ArtistId, Name, COUNT(TrackId) as track_count \nFROM Track \nGROUP BY ArtistId \nORDER BY track_count DESC \nLIMIT 5 \n;"}'}
                            """
                            json_str2 = json_str.replace("\'", "\"\"\"")
                            json_dict = ast.literal_eval(json_str2)
                        except SyntaxError as e:
                            debug_msg(f"process_finish_msg Error parsing json_str2 {json_str2} with error {e}")
                            # There is no way to parse this, so just skip
                            continue
                    
                    # Sanity check on json_dict
                    if 'name' not in json_dict or 'arguments' not in json_dict:
                        debug_msg(f"process_finish_msg json_dict does not have name or arguments {json_dict}")
                        continue

                    """
                    https://stackoverflow.com/questions/22394235/invalid-control-character-with-python-json-loads
                    Escape control chars in arguments to pass json.load(text, strict=True) when deserializing on the client side
                    """
                    def excape_crtl_chars(text: str)->str:
                        # Define a dictionary that maps control characters to their escape sequences
                        control_chars_dict = {i: '\\u{:04x}'.format(i) for i in range(32)}
                        # Add some additional control characters that aren't in the default translation table
                        control_chars_dict[127] = '\\u007f'
                        # Remove control characters from the string
                        cleaned_text = text.translate(control_chars_dict)
                        if len(cleaned_text) != len(text):
                            debug_msg(f"process_finish_msg escape control chars from {text} to {cleaned_text}")
                        return cleaned_text
                    
                    json_dict['name'] = excape_crtl_chars(json_dict['name'])
                    json_dict['arguments'] = excape_crtl_chars(json_dict['arguments'])

                    # Gen openai like call id with 24 random characters
                    def gen_openai_like_call_id()->str:
                        import random
                        import string
                        length = 24
                        charset = string.ascii_lowercase + string.digits 
                        return ''.join(random.choice(charset) for i in range(length))
                    
                    response = FunctionCallResponse(
                        id=f'call_{gen_openai_like_call_id()}',
                        function=FunctionNameArg(name=json_dict['name'], arguments=json_dict['arguments'])
                        )
                    
                    finish_response = response.model_dump(mode='json')                    
                    debug_msg(f"process_finish_msg response:\n{finish_response}")
                    finish_responses.append(finish_response)
                
        except Exception as e:
            debug_msg(f"process_finish_msg exception: {e}")
            exception_occurred = True

        return finish_responses, exception_occurred
