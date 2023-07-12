'''
Based on
https://github.com/xorbitsai/inference
'''

from xinference.client import Client

class XinferenceModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, model_name, model_uid, endpoint):
        assert isinstance(model_uid, str)
        assert isinstance(endpoint, str)

        self.endpoint = endpoint
        client = Client(endpoint)
        self.model_uid = model_uid
        self.model = client.get_model(self.model_uid)
        return self, self
