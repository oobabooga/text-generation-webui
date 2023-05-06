import random
import string
from ..base import PreparedPrompt, Source
from ..source import TextSource, UrlSource
from bs4 import BeautifulSoup
from urlextract import URLExtract

class SourceBuilder():
    def __init__(self) -> None:
        pass
    
    def from_text(self, text: str) -> Source:
        return self.infer(text)
    
    def infer(self, string: str) -> Source:
        inferred_source = None
        extractor = URLExtract()
        
        if extractor.has_urls(string):
            inferred_source = UrlSource(string)
            soup = BeautifulSoup(inferred_source.get(), features="html.parser")
            metas = soup.find_all('meta')
            descriptions = [meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description']
            # Todo - page title, index page by title semantic search by page name
            # titles = [meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'title']
            injection_point_name = ''
            
            injection_point_name = self.get_random_short_hash()
                
            inferred_source.metadata.add('inferred_injection_point_name', injection_point_name)
            inferred_source.metadata.add('descriptions', descriptions)
            inferred_source.metadata.add('inferred_from', string)
            inferred_source.name = injection_point_name
        
        else:
            inferred_source = TextSource(string)
            
        return inferred_source
    
    def get_hollow_injection_points(self, prepared_prompt: PreparedPrompt) -> list[str]:
        extractor = URLExtract()
        hollow_injection_points = []
        urls = extractor.find_urls(prepared_prompt.source_prompt)
        if len(urls) > 0:
            for url in urls:
                hollow_injection_points.append(url)
        
        return hollow_injection_points
            
            
    def get_random_short_hash(self) -> str:
        alphabet = string.ascii_lowercase + string.digits
        return ''.join(random.choices(alphabet, k=8))
            
        