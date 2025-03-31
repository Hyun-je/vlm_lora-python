from functools import partial
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image

from .google_gemma_3 import collate_fn


def load(**kwargs):

    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-pt",
        **kwargs
    )

    processor = AutoProcessor.from_pretrained(
        "google/gemma-3-4b-it",
        use_fast=True,
        cache_dir=kwargs["cache_dir"]
    )

    return model, processor, partial(collate_fn, processor=processor)