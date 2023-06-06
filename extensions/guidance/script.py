from extensions.guidance import guidance_server
from modules import shared

def setup():
    guidance_server.start_server(shared.args.guidance_port)
