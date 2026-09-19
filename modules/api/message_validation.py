from .errors import InvalidRequestError


SUPPORTED_CHAT_ROLES = ('system', 'developer', 'user', 'assistant', 'tool')


def validate_chat_messages(messages):
    for message in messages:
        if 'role' not in message:
            raise InvalidRequestError(message="messages: missing role", param='messages')

        role = message['role']
        if role not in SUPPORTED_CHAT_ROLES:
            raise InvalidRequestError(message=f"role: {role} is not supported.", param='messages')

        content = message.get('content')
        if content is None:
            # OpenAI allows content: null on assistant messages when tool_calls is present
            if role == 'assistant' and message.get('tool_calls'):
                message['content'] = ''
            else:
                raise InvalidRequestError(message="messages: missing content", param='messages')

        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict) or 'type' not in item:
                raise InvalidRequestError(message="messages: invalid content item format", param='messages')

            if item['type'] not in ('text', 'image_url'):
                raise InvalidRequestError(message="messages: unsupported content type", param='messages')

            if item['type'] == 'text':
                if 'text' not in item:
                    raise InvalidRequestError(message="messages: missing text in content item", param='messages')
                if not isinstance(item['text'], str):
                    raise InvalidRequestError(message="messages: text content must be a string", param='messages')

            if item['type'] == 'image_url':
                if 'image_url' not in item or not isinstance(item['image_url'], dict) or 'url' not in item['image_url']:
                    raise InvalidRequestError(message="messages: missing image_url in content item", param='messages')
                if not isinstance(item['image_url']['url'], str):
                    raise InvalidRequestError(message="messages: image_url must contain a string URL", param='messages')
