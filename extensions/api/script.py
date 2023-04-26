import extensions.api.blocking_api as blocking_api
import extensions.api.streaming_api as streaming_api
from modules import shared

BLOCKING_PORT = 5000
STREAMING_PORT = 5005

def setup():
    blocking_api.start_server(BLOCKING_PORT, share=shared.args.public_api)
    streaming_api.start_server(STREAMING_PORT, share=shared.args.public_api)
