'''
Based on
https://github.com/xorbitsai/inference
'''


class XinferenceModel:
    def __init__(self):
        try:
            from xinference.client import Client
        except ImportError:
            error_message = "Failed to import module 'xinference'"
            installation_guide = [
                "Please make sure 'xinference' is installed. ",
                "You can visit the original git repo for details:\n",
                "https://github.com/xorbitsai/inference",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")


    @classmethod
    def from_pretrained(self, model_name, model_uid, endpoint):
        assert isinstance(model_uid, str)
        assert isinstance(endpoint, str)

        self.endpoint = endpoint
        client = Client(endpoint)
        self.model_uid = model_uid
        self.model = client.get_model(self.model_uid)
        return self, self
