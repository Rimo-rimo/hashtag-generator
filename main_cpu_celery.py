# Image to Hashtags✨ by Rimo, v1.0.0
"""
Get the Hashtags from image uploaded

Usage:
    $ uvicorn main_cpu_celery:app --reload --host=0.0.0.0 --port=30000

Models: 
    https://huggingface.co/MAGAer13/mplug-owl-llama-7b
"""

import warnings
warnings.filterwarnings(action="ignore")
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch

from celery_worker import get_hashtag
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, validator
from typing import Optional
import base64

# Basic Models
def basic_mplug_owl(pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b', device='cuda'):
    model = MplugOwlForConditionalGeneration.from_pretrained(
                pretrained_ckpt,
                torch_dtype=torch.bfloat16)
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model.to(device)
    return model, tokenizer, processor

# Multilingual_Models
def multilingual_mplug_owl(pretrained_ckpt='MAGAer13/mplug-owl-bloomz-7b-multilingual', device='cuda'):
    model = MplugOwlForConditionalGeneration.from_pretrained(
                pretrained_ckpt,
                torch_dtype=torch.bfloat16)
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model.to(device)
    return model, tokenizer, processor

def post_process(result):
    result = result.replace(" ","")
    result = result.replace(",","")
    result = result.replace(".","")
    result = result.replace("\n","")
    hashtags = []
    hashtag = ""
    for i in result:
        if i == "#":
            hashtags.append(hashtag)
            hashtag = i
        else:
            if i in "1234567890":
                pass
            else:
                hashtag += i
    return [i[1:] for i in hashtags[1:]]

# class Base64Bytes(bytes):
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate

#     @classmethod
#     def validate(cls, v):
#         try:
#             return base64.b64decode(v)
#         except Exception:
#             raise ValueError('Invalid base64-encoded string')

# Setting
device = "cpu"
model, tokenizer, processor = basic_mplug_owl(pretrained_ckpt='MAGAer13/mplug-owl-llama-7b', device=device)
prompt = "Please create 7 hashtags for Instagram through a sentence that summarizes the image in one sentence."
generate_kwargs = {
    'max_length': 100,
    'do_sample' : False,
    'early_stopping' : True,
    'num_beams' : 1,
    'temperature' : 1.0,
    'top_k': 50,
    'top_p': 1.0,
    'repetition_penalty' : 1.0,
    'no_repeat_ngram_size' : 0,
    'num_return_sequences' : 1,
    'use_cache' : True,
}
print("=============get ready===========")

# Deploy
app = FastAPI()

# class Item(BaseModel):
#     image : Base64Bytes
#     meta_data : Optional[dict] = None

@app.get("/")
def root():
    return "Hashtag-Generator"

@app.post("/hashtags")
def generate_tag(image:bytes = File(...), meta_data:Optional[dict] = None):
    result = get_hashtag(image, meta_data, prompt, generate_kwargs, tokenizer, processor, model)
    hashtags = post_process(result)
    return hashtags

@app.post("/gpt_hashtags")
def generate_tag_from_gpt(image:bytes = File(...), meta_data:Optional[dict] = None):
    result = get_hashtag(image, meta_data, prompt, generate_kwargs, tokenizer, processor, model)
    hashtags = post_process(result)
    return hashtags
    