from __future__ import annotations
from chunkers import BaseChunker
from embedder import Embedder
from vectorstore import ChromaDBStore
from model.search_result import SearchResult
from openai import AzureOpenAI
import config

class RAG:
    def __init__(self, embedder: Embedder, vector_store: ChromaDBStore, chunker: BaseChunker | None = None):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.client = AzureOpenAI(
                api_key         = config.API_KEY,
                api_version     = config.API_VERSION,
                azure_endpoint  = config.DIAL_URL
            )

    def get_context(self, query: str, n_results: int = 3, where: dict | None = None) -> tuple[str, list[SearchResult]]:
        query_embedding = self.embedder.embed_query(query)
        results: list[SearchResult] = self.vector_store.search(
            query_embedding, n_results=n_results, where=where
        )
        context = "\n\n---\n\n".join(r.text for r in results)
        return context, results

    def get_answer(self, query: str, n_results: int = 3, where: dict | None = None) -> tuple[str, list[SearchResult]]:
        context, results = self.get_context(query, n_results=n_results, where=where)
        response = self.client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that answers questions about Donald Trump's tweets and posts. "
                    "You are given excerpts from his tweets as context. "
                    "Synthesize the relevant information from the context to answer the question. "
                    "Quote or paraphrase specific tweets where helpful. "
                    "If the context contains partial or indirect information, use it — do not refuse just because the answer is incomplete. "
                    "Only say you cannot answer if the context contains absolutely no relevant information."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content, results
