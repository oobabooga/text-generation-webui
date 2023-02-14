from nataili_blip.model_manager import BlipModelManager
from nataili_blip.caption import Caption

def load_model():
    model_name = "BLIP"
    mm = BlipModelManager()
    mm.download_model(model_name)
    mm.load_blip(model_name)
    return Caption(mm.loaded_models[model_name]["model"], mm.loaded_models[model_name]["device"])