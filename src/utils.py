from glob import glob
import math
import typing
import logging

import torch
from peft import PeftModel

def get_wiki_filepath(data_dir):
    return glob(f"{data_dir}/*/wiki_*")


def wiki_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    split_size = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    # end_idx = min((worker_id+1) * split_size, len(dataset.data))
    dataset.start = overall_start + worker_id * split_size
    dataset.end = min(dataset.start + split_size, overall_end)  # index error 방지


def get_passage_file(p_id_list: typing.List[int]) -> str:
    """passage id를 받아서 해당되는 파일 이름을 반환합니다."""
    target_file = None
    p_id_max = max(p_id_list)
    p_id_min = min(p_id_list)
    for f in glob("../data/processed_passages/*.p"):
        s, e = f.split("/")[-1].split(".")[0].split("-")
        s, e = int(s), int(e)
        if p_id_min >= s and p_id_max <= e:
            target_file = f
    if target_file is None:
        logging.debug(f"No file found for passage IDs: {p_id_list}")
    return target_file

def apply_lora(model, adaptor_path):
    lora_model = PeftModel.from_pretrained(model, adaptor_path)
    return lora_model