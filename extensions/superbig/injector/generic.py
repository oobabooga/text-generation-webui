from typing import Tuple
from ..searcher import CosineSimilaritySearcher
from ..source import SourceBuilder
from ..base import Bucket, Collecter, Chunker, InjectionPoint, Injector, Page, PreparedPrompt, Source, Embedder, Chunk, Window
from ..prepared_prompt import AlpacaPreparedPrompt, GenericPreparedPrompt
from ..metadata import MetadataBuilder
import re

class GenericInjector(Injector):
    """
    Prepares prompts, chunks data sources, collects them into DBs, and injects them back into prompts
    """
    
    def __init__(self, chunker: Chunker, collector: Collecter, embedder: Embedder, sources: dict[InjectionPoint, Source]):
        self.chunker = chunker
        self.collector = collector
        self.sources = sources
        self.inferred_source_mappings = {}
        self.embedder = embedder
        self.prepared_output = ''
        self.metadata_builder = MetadataBuilder()
        self.source_builder = SourceBuilder()
        self.searcher = CosineSimilaritySearcher()
        self.hollow_injection_points = {}
    
    def get_prepared_prompt(self, text: str) -> PreparedPrompt:
        prepared_prompt = self.get_inferred(text)
        prepared_prompt.from_prompt(text)
        return prepared_prompt
        
    def get_inferred(self, text: str) -> PreparedPrompt:
        if(text.find('### Instruction') != 1):
            return AlpacaPreparedPrompt()
        else:
            return GenericPreparedPrompt()
        
    def add_and_infer_hollow_injection_points(self, hollow_injection_points: list[str]) -> list[InjectionPoint]:
        real_injection_points = []
        for hollow_injection_point in hollow_injection_points:
            if hollow_injection_point not in self.hollow_injection_points:
                real_injection_point = self.add_generic_source(hollow_injection_point)
                self.hollow_injection_points[hollow_injection_point] = real_injection_point
            real_injection_points.append(self.hollow_injection_points[hollow_injection_point])
            
        return real_injection_points
    
    def add_generic_source(self, text: str) -> Source:
        source = self.source_builder.from_text(text)
        source_name = source.metadata.get('inferred_injection_point_name')
        inferred_from = source.metadata.get('inferred_from')
        self.inferred_source_mappings[inferred_from] = source
        injection_point = InjectionPoint(source_name)
        injection_point.target = text
        self.add_source(injection_point, source)
        return injection_point
    
    def add_source(self, injection_point: InjectionPoint, source: Source):
        self.sources[injection_point] = source   
        
    def infer_source_from_injection_point(self, injection_point: InjectionPoint) -> Source:
        return self.sources[injection_point]
        
    def load_and_cache(self, injection_points: list[InjectionPoint]) -> list[Bucket]:
        all_buckets = []
        for injection_point in injection_points:
            real_source = self.infer_source_from_injection_point(injection_point)
            if real_source.chunked:
                continue
            
            print(real_source.name, " is not chunked. Chunking it now...")
            
            loaded_data = real_source.get()
            data_chunks = self.chunker.make_chunks(loaded_data)
            bucket = self.make_bucket(data_chunks, injection_point)
            real_source.chunked = True
            all_buckets.append(bucket)
        
        print('Adding ', len(all_buckets), ' collections')
        self.collector.add(all_buckets)
    
    def make_bucket(self, chunks: list[Chunk], injection_point: InjectionPoint):
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for idx, chunk in enumerate(chunks):
            chunk.embeddings = self.embedder.embed(chunk.text)
            chunk.id = f"id{idx}"
            chunk = self.metadata_builder.enrich(chunk)
            ids.append(chunk.id)
            embeddings.append(chunk.embeddings)
            metadatas.append(chunk.metadatas)
            documents.append(chunk.text)
            
        bucket = Bucket(injection_point.real_name, chunks)
        bucket.ids = ids
        bucket.embeddings = embeddings
        bucket.metadatas = metadatas
        bucket.documents = documents
        
        return bucket
    
    def prepare(self, text: str) -> PreparedPrompt:
        print('Preparing prompt...')
        prepared_prompt: PreparedPrompt = self.get_prepared_prompt(text)
        print('Getting injections...')
        injection_points = prepared_prompt.get_injection_points()
        hollow_injection_points = self.source_builder.get_hollow_injection_points(prepared_prompt)
        print('Inferring injections...')
        injection_points += self.add_and_infer_hollow_injection_points(hollow_injection_points)
        print('Loading and caching injections...')
        self.load_and_cache(injection_points)
        return prepared_prompt
    
    def choose_best_source(self, prepared_prompt: PreparedPrompt) -> list[Tuple[InjectionPoint, Source]]:
        source_description_embeddings = [self.embedder.embed(source.metadata.get('description')) for _, source in self.sources.items()]
        search_string_embeddings = [self.embedder.embed(search_string) for search_string in prepared_prompt.get_search_strings()]
        results = self.searcher.search(search_string_embeddings, source_description_embeddings)
        results = [list(self.sources.items())[result] for result in results]
        return results
    
    def parse_injection_levels(self, string: str) -> Tuple[bool, list[int]]:
        parts = string.split(':')
        is_expansion = False
        
        if "+" in parts[1]:
            is_expansion = True
            parts[1] = parts[1].replace('+', '')
            
        return (is_expansion, [int(parts[1])] * int(parts[0]))
    
    def get_relevant_context(self, search_strings: list[str], injection_levels: list[int] | str, injection_point: InjectionPoint, additional_information: str):
        optimized_context = []
        previous_relevant_context = search_strings
        search_results_page = Page(additional_information)
        is_expansion = False
        collected_chunks = []
        
        if isinstance(injection_levels, str):
            is_expansion, injection_levels = self.parse_injection_levels(injection_levels)
             
        for injection_level in injection_levels:
            relevant_chunks = self.collector.get_chunks(previous_relevant_context, injection_level, injection_point, exclude_chunks=collected_chunks)[injection_point]
            search_results_page.add_chunks(relevant_chunks)
            collected_chunks.extend(relevant_chunks)
            relevant_portion = [chunk.text for chunk in relevant_chunks]
            optimized_context.append(search_results_page)
            
            if is_expansion:
                previous_relevant_context += relevant_portion
            else:
                previous_relevant_context = relevant_portion
                
            search_results_page = Page('More information:')
            
        results_window = Window(optimized_context)
            
        return {injection_point: results_window}
        
    def inject(self, prepared_prompt: PreparedPrompt) -> str:
        print('Choosing the best information source...')
        best_source_injection_point, best_source = self.choose_best_source(prepared_prompt)[0]
        print("The best source seems to be ", best_source_injection_point.target)
        print("Searching...")
        relevant_context = self.get_relevant_context(prepared_prompt.get_search_strings(), "10:3+", best_source_injection_point, best_source.metadata.get('description'))
        for injection_point, data in relevant_context.items():
            prepared_prompt.inject(injection_point, data.view())
        print("Injecting...")
        prompt = prepared_prompt.rebuild()
        return prompt