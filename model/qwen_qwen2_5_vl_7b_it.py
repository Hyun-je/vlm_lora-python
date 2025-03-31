from functools import partial
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from .qwen_qwen2_5_vl import collate_fn


def load(**kwargs):

    kwargs["use_cache"]=False,

    # Load model and tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        **kwargs
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast=True,
        cache_dir=kwargs["cache_dir"]
    )

    return model, processor, partial(collate_fn, processor=processor)