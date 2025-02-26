import argparse
import datetime
import os
import random
from dotenv import load_dotenv

import numpy as np
import pytz
import torch
import torch.backends.cudnn as cudnn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig, \
    AutoConfig

import retrieval
from config import Config
from retrieve import retrieve

def parse_args():
    parser = argparse.ArgumentParser(description="retrieval parameters")
    parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--question", type=str, required=True, help="ask question to LLM")
    return parser.parse_args()

def main():
    load_dotenv()
    
    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    llm_config = cfg.config.llm
    retrieval_config = cfg.config.retriever
    data_config = cfg.config.datasets

    question = args.question
    DEVICE = run_config.device
    
    quant_config = None
    if llm_config._4_bit_quant:
        quant_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
    if llm_config._8_bit_quant:
        quant_config=BitsAndBytesConfig(
                load_in_8bit=True,
            )

    config = AutoConfig.from_pretrained(llm_config.llm_path)
    model = AutoModelForCausalLM.from_pretrained(
            llm_config.llm_path,
            config=config,
            torch_dtype=torch.float32 if not isinstance(quant_config, BitsAndBytesConfig) else None,
            trust_remote_code=True,
            quantization_config=quant_config if isinstance(quant_config, BitsAndBytesConfig) else None,
        ).to(DEVICE)
    
    if llm_config.lora:
        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            inference_mode=False,
                            r=llm_config.lora_rank,
                            lora_alpha=llm_config.lora_alpha,
                            lora_dropout=llm_config.lora_dropout,
                            target_modules = ["q_proj", "v_proj", "gate_proj"]
                        )
        model = get_peft_model(model, peft_config)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        llm_config.llm_path,
        trust_remote_code=True,
    )

    retriever = None
    if retrieval_config.step_1_model == "Semantic":
        if retrieval_config.use_Faiss:
            step_1_retriever = getattr(retrieval, retrieval_config.step_1_model)(
                dense_model_name=retrieval_config.semantic_embeder,
                index_output_path=retrieval_config.faiss_index_output_path,
                chunked_path=retrieval_config.faiss_chunk_path,
                data_path=data_config.data_path,
                context_path=data_config.context_path,
                FAISS=retrieval_config.use_Faiss,
                device=DEVICE
            )
            step_1_retriever.get_dense_embedding_with_faiss()
        else:
            step_1_retriever = getattr(retrieval, retrieval_config.step_1_model)(
                dense_model_name=retrieval_config.semantic_embeder,
                data_path=data_config.data_path,
                context_path=data_config.context_path,
                device=DEVICE
            )
            step_1_retriever.get_dense_embedding()
    elif retrieval_config.step_1_model == "Syntactic":
        step_1_retriever = getattr(retrieval, retrieval_config.step_1_model)(
            tokenize_fn=tokenizer,
            vectorizer_type=retrieval_config.syntactic_embeder,
            data_path=data_config.data_path,
            context_path=data_config.context_path
        )
        step_1_retriever.get_sparse_embedding()
    elif retrieval_config.step_1_model == "Elastic":
        step_1_retriever = getattr(retrieval, retrieval_config.step_1_model)(
            data_path=data_config.data_path,
            context_path=data_config.context_path,
            device=DEVICE
        )
        if retrieval_config.elastic_type == "sparse":
            step_1_retriever.get_sparse_embedding_with_elastic()
        elif retrieval_config.elastic_type == "dense":
            step_1_retriever.get_dense_embedding_with_elastic()

    if retrieval_config.step_2_model:
        if retrieval_config.step_2_model == "Semantic":
            if retrieval_config.use_Faiss:
                step_2_retriever = getattr(retrieval, retrieval_config.step_2_model)(
                    dense_model_name=retrieval_config.semantic_embeder,
                    index_output_path=retrieval_config.faiss_index_output_path,
                    chunked_path=retrieval_config.faiss_chunk_path,
                    data_path=data_config.data_path,
                    context_path=data_config.context_path,
                    FAISS=retrieval_config.use_Faiss,
                    device=DEVICE
                )
                step_2_retriever.get_dense_embedding_with_faiss()
            else:
                step_2_retriever = getattr(retrieval, retrieval_config.step_2_model)(
                    dense_model_name=retrieval_config.semantic_embeder,
                    data_path=data_config.data_path,
                    context_path=data_config.context_path,
                    device=DEVICE
                )
                step_2_retriever.get_dense_embedding()
        elif retrieval_config.step_2_model == "Syntactic":
            step_2_retriever = getattr(retrieval, retrieval_config.step_2_model)(
                tokenize_fn=tokenizer,
                vectorizer_type=retrieval_config.syntactic_embeder,
                data_path=data_config.data_path,
                context_path=data_config.context_path
            )
            step_2_retriever.get_sparse_embedding()
        elif retrieval_config.step_2_model == "Elastic":
            step_2_retriever = getattr(retrieval, retrieval_config.step_2_model)(
                data_path=data_config.data_path,
                context_path=data_config.context_path,
                device=DEVICE
            )
            if retrieval_config.elastic_type == "sparse":
                step_2_retriever.get_sparse_embedding_with_elastic()
            elif retrieval_config.elastic_type == "dense":
                step_2_retriever.get_dense_embedding_with_elastic()

    if retrieval_config.rerank:
        retriever = getattr(retrieval, "Reranker")(
            step1_model=step_1_retriever,
            step2_model=step_2_retriever
        )
    elif retrieval_config.hybrid:
        retriever = getattr(retrieval, "HybridSearch")(
            tokenize_fn=tokenizer,
            step1_model=step_1_retriever,
            step2_model=step_2_retriever
        )
    
    if retriever is None:
        retriever = step_1_retriever
    
    reference = retrieve(retriever, model, tokenizer, question, llm_config.max_txt_len, cfg)
    llm_prompt = llm_config.prompt_template.format(
        question=question,
        reference=reference
    )
    inputs = tokenizer(llm_prompt, return_tensors="pt")

    outputs = model.generate(
        inputs['input_ids'].to(DEVICE),
        max_new_tokens=llm_config.max_txt_len,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    

if __name__=="__main__":
    main()