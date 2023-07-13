'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

from xinference.client import Client


class XinferenceModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, model_name, model_uid, endpoint):
        self.endpoint = int(endpoint)
        client = Client(f"http://localhost:{self.endpoint}")

        if any(name in model_name for name in ['wizardlm', 'Wizardlm']):
            model_name = 'wizardlm-v1.0'

        if int(model_uid) == 0:
            model_uid = client.launch_model(model_name=model_name)
        self.model_uid = model_uid
        self.model = client.get_model(self.model_uid)
        return self, self
