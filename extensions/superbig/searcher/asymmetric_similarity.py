from ..base import Embedding, SearchResult, Searcher

class AsymmetricSimilaritySearcher(Searcher):
    def search(self, query: list[str] | list[Embedding], over_input: list[str] | list[Embedding]) -> list[SearchResult]:
        ...