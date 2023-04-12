import asyncio
import json
import websockets
import sys
def printonce(message):
    if not hasattr(printonce, "printed"):
        printonce.printed = set()
    if message not in printonce.printed:
        print(message)
        printonce.printed.add(message)

import base64
import getpass
import websockets

async def connect():
    uri = "wss://example.com:5001/wsapi"
    username = ""
    password = ""
    is_authenticated = False
    while not is_authenticated:
        try:
            headers = {}
            if username and password:
                headers['Authorization'] = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                is_authenticated = True
                await exe_websocket(websocket)
                pass
        except websockets.exceptions.InvalidHandshake as e:
            if isinstance(e, websockets.exceptions.InvalidStatusCode) and e.status_code == 401:
                if not username and not password:
                    headers = {k.lower(): v for k, v in dict(e.headers).items()}
                    if 'www-authenticate' in headers and 'asic' in headers['www-authenticate']:
                        print("This server requires authentication.")
                        username = input("Enter your username: ")
                        password = getpass.getpass("Enter your password: ")
                    else:
                        print("Server is not configured with basic authentication.")
                        break
                else:
                    print("Authentication failed. Please try again.")
                    username = input("Enter your username: ")
                    password = getpass.getpass("Enter your password: ")
            else:
                raise e

async def exe_websocket(websocket):
    prompt = """Once upon a time"""
    generate_params = {
        'max_new_tokens': 1800,
        'temperature': 0.8,
        'top_p': 0.9,
        'rep_pen': 1.1
    }
    stopping_strings = ["\nYou", "\nYou:", "You:"]
    
    data = {
        'prompt': prompt,
        **generate_params,
        'stopping_strings': stopping_strings
    }

    await websocket.send(json.dumps(data))
    print(prompt, end="")
    
    current_prompt = prompt
    while True:
        try:
            response = await websocket.recv()
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WS connection closed while recv(): {str(e)}")
            break
        except websockets.exceptions.ConnectionClosedOK:
            break
        parsed_response = json.loads(response)

        if "generation_complete" in parsed_response and parsed_response["generation_complete"]:
            print("\n\nServer: generation_complete\n")
            break

        if "text" in parsed_response:
            full_text = parsed_response["text"]
            new_tokens = full_text[len(current_prompt):]
            print(new_tokens, end="")
            sys.stdout.flush()
            current_prompt = full_text
        elif "error" in parsed_response:
            print(f"Error: {parsed_response['error']}")

asyncio.run(connect())
