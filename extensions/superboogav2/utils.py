"""
This module contains common functions across multiple other modules.
"""

from extensions.superboogav2 import parameters


# Create the context using the prefix + data_separator + postfix from parameters.
def create_context_text(results):
    context = parameters.get_prefix() + parameters.get_data_separator().join(results) + parameters.get_postfix()

    return context


# Create metadata with the specified source
def create_metadata_source(source: str):
    return {'source': source}
