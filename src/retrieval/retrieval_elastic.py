import torch
import os
import sys
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indexer import indexers
from indexer import IndexRunner
from .retrieval import Retrieval 

class Elastic(Retrieval):
    def __init__(
        self,
        data_path: Optional[str] = "../../data/",
        context_path: Optional[str] = "wiki_documents_original.csv",
        es_host: str = "localhost",
        es_port: int = 9200,
        es_scheme: str = "http",
        index_name: str = "documents",
        model_name: str = "BAAI/bge-m3",
        indexer_type: str = "ElasticIndexer",
        device = None
    ):  
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.data_path = data_path
        self.context_path = context_path
        self.es_host = es_host
        self.es_port = es_port
        self.es_scheme = es_scheme
        self.index_name = index_name
        self.indexer_type = indexer_type

        self.indexer = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def get_sparse_embedding_with_elastic(self, contexts=None):
        self.indexer = getattr(indexers, self.indexer_type)()
        IndexRunner(
                encoder=self.encoder,
                tokenizer=self.tokenizer,
                data_dir=os.path.join(self.data_path, self.context_path),
                indexer_type=self.indexer_type,
                indexer=self.indexer,
                use_elastic=True,
                contexts=contexts
            ).run()

    def get_dense_embedding_with_elastic(self):
        pass

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        query_body = {
            "match": {
                "content": query
            }
        }
        response = self.indexer.es.search(index=self.index_name, query=query_body, size=k)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"]
            })
        return results

if __name__ == "__main__":
    pass
