import time

import extensions.api.blocking_api as blocking_api
import extensions.api.streaming_api as streaming_api
from modules import shared
from modules.logging_colors import logger


def setup():
    logger.warning("\nThe current API is deprecated and will be replaced with the OpenAI compatible API on November 13th.\nTo test the new API, use \"--extensions openai\" instead of \"--api\".\nFor documentation on the new API, consult:\nhttps://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API")
    blocking_api.start_server(shared.args.api_blocking_port, share=shared.args.public_api, tunnel_id=shared.args.public_api_id)
    if shared.args.public_api:
        time.sleep(5)

    streaming_api.start_server(shared.args.api_streaming_port, share=shared.args.public_api, tunnel_id=shared.args.public_api_id)
