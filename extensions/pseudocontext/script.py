from extensions.pseudocontext.instruct_models import InstructChunker, make_instruct_provider
from .provider import PseudocontextProvider

provider = PseudocontextProvider()
instruct_provider = make_instruct_provider()

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    return instruct_provider.with_pseudocontext(string)

def ui():
    pass