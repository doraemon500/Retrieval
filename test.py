from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig, \
    AutoConfig
from rank_bm25 import BM25Plus

tokenized_corpus = ["sample", "another example document" ,"more ndata or bm25"]


tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
    )

tokenized_corpus = tokenizer(tokenized_corpus)['input_ids']
print(tokenized_corpus)

bm25 = BM25Plus(tokenized_corpus)


query = "sample document"
query = tokenizer(query)['input_ids']
print(query)

# 쿼리에 대한 점수 계산
scores = bm25.get_scores(query)
print(scores)