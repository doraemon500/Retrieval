import logging
import os
import time

import pandas as pd
import torch
from contextlib import contextmanager
from datasets import Dataset
from typing import  Optional, NoReturn
from tqdm.auto import tqdm

from .retrieval import HybridSearch
from .retrieval_syntactic import Syntactic
from .retrieval_semantic import Semantic
from .retrieval import Retrieval
from utils import set_seed

set_seed(2024)
torch.use_deterministic_algorithms(False)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")


class Reranker(Retrieval):
    def __init__(
        self,
        step1_model: Retrieval,
        step2_model: Retrieval,
        data_path: Optional[str] = "../../data/",
        context_path: Optional[str] = "wiki_documents_original.csv",
    ) -> NoReturn:
        self.step1_model = step1_model
        self.step2_model = step2_model

        self.data_path = data_path
        if isinstance(self.step1_model, Syntactic):
            self.contexts = self.step1_model.contexts
        elif isinstance(self.step2_model, Syntactic):
            self.contexts = self.step2_model.contexts
        else:
            with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
                wiki = pd.read_csv(f)
            self.contexts = list(dict.fromkeys(wiki['content']))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    def _step1_retrieve(self, queries, topk: Optional[int] = 1, alpha: Optional[int]=0.5):
        if isinstance(self.step1_model, HybridSearch):
            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step1_model.retrieve(queries, topk=topk, alpha=alpha)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step1_model.retrieve(queries, topk=topk, alpha=alpha)
                return datas
        else:
            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step1_model.retrieve(queries, topk=topk)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step1_model.retrieve(queries, topk=topk)
                return datas
    
    def _step2_retrieve(self, queries, topk: Optional[int] = 1, alpha: Optional[int]=0.5):
        if isinstance(self.step2_model, HybridSearch):
            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step2_model.retrieve(queries, topk=topk, alpha=alpha)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step2_model.retrieve(queries, topk=topk, alpha=alpha)
                return datas
        else:
            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step2_model.retrieve(queries, topk=topk)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step2_model.retrieve(queries, topk=topk)
                return datas

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1, alpha_1: Optional[int] = 0, alpha_2: Optional[int] = 0):
        retrieved_contexts = []
        if isinstance(query_or_dataset, str):
            _, doc_contexts = self._step1_retrieve(query_or_dataset, topk, alpha=alpha_1)
            retrieved_contexts = doc_contexts
        elif isinstance(query_or_dataset, Dataset):
            for _, example in enumerate(tqdm(query_or_dataset, desc="[Rerank first retrieval]: ")):
                doc_contexts = self._step1_retrieve(example['question'], topk, alpha=alpha_1)
                retrieved_contexts.append(doc_contexts)

        half_topk = int(topk / 2)

        if isinstance(query_or_dataset, str):
            _, second_datas = self._step2_retrieve(query_or_dataset, half_topk, contexts=retrieved_contexts, alpha=alpha_2)
            return second_datas
        elif isinstance(query_or_dataset, Dataset):
            second_datas = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Rerank second retrieval] ")):
                context = retrieved_contexts[idx]
                doc_contexts = self._step2_retrieve(example['question'], half_topk, contexts=context, alpha=alpha_2)
                template = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(doc_contexts),
                }
                second_datas.append(template)
            second_datas = pd.DataFrame(second_datas)
            return second_datas

if __name__ == "__main__":
    pass
