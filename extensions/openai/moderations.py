
def moderations(input):

    # input, string or array
    # for now do nothing, just don't error.
    return {
        "id": "modr-5MWoLO",
        "model": "text-moderation-001",
        "results": [{
            "categories": {
                "hate": False,
                "hate/threatening": False,
                "self-harm": False,
                "sexual": False,
                "sexual/minors": False,
                "violence": False,
                "violence/graphic": False
            },
            "category_scores": {
                "hate": 0.0,
                "hate/threatening": 0.0,
                "self-harm": 0.0,
                "sexual": 0.0,
                "sexual/minors": 0.0,
                "violence": 0.0,
                "violence/graphic": 0.0
            },
            "flagged": False
        }]
    }