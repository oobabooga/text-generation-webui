import time

import extensions.api.blocking_api as blocking_api
import extensions.api.streaming_api as streaming_api
from modules import shared


def setup():
    blocking_api.start_server(shared.args.api_blocking_port, share=shared.args.public_api, tunnel_id=shared.args.public_api_id)
    if shared.args.public_api:
        time.sleep(5)

    streaming_api.start_server(shared.args.api_streaming_port, share=shared.args.public_api, tunnel_id=shared.args.public_api_id)
