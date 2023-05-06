import requests
from ..base import Source
from lxml.html.clean import Cleaner
import unicodedata

class UrlSource(Source):
    def __init__(self, url=''):
        super().__init__()
        
        if len(url) > 1:
            self.set(url)
        
    def get(self) -> str:
        if self.contents is not None:
            return self.contents
        
        response = requests.get(self.url)
        
        if response.status_code != 200:
            return ''
        
        self.contents = self.sanitize(unicodedata.normalize('NFKC', response.text))
        return super().get()
            
    def set(self, url: str):
        self.url = url
        super().set()
        
    def sanitize(self, dirty_html):
        cleaner = Cleaner(page_structure=True,
                    meta=True,
                    embedded=True,
                    links=True,
                    style=True,
                    processing_instructions=True,
                    inline_style=True,
                    scripts=True,
                    javascript=True,
                    comments=True,
                    frames=True,
                    forms=True,
                    annoying_tags=True,
                    remove_unknown_tags=True,
                    safe_attrs_only=True,
                    safe_attrs=frozenset(['src','color', 'href', 'title', 'class', 'name', 'id']),
                    remove_tags=('span', 'font', 'div', 'a', 'img')
                    )
        clean = cleaner.clean_html(dirty_html)
        return clean.replace('\t', '').replace('\r','')