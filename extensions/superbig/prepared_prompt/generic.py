from ..base import PreparedPrompt
import re

class GenericPreparedPrompt(PreparedPrompt):
    """
    Format Generic prompts
    """

    def from_prompt(self, prompt: str) -> dict:
        pattern = r'\[\[\[.*?\]\]\]'
        prompt_without_injection_points = re.sub(pattern, '', prompt)
        
        return super().from_prompt(prompt, {
            'prompt': prompt_without_injection_points
        })