"""
[Command]
celery -A celery_worker worker --loglevel=info --concurrency=1
"""
import warnings
warnings.filterwarnings(action="ignore")
import torch
from PIL import Image
from celery import Celery
import io

# Setting
celery_task = Celery(
    'app',
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0"
)

@celery_task.task
def get_hashtag(image, meta_data, prompt, generate_kwargs, tokenizer, processor, model):
    # Prompt
    prompts = [
    f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <image>
    Human: {prompt}
    AI: ''']

    # Preprocessing
    images = [Image.open(io.BytesIO(image))]
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

    return sentence