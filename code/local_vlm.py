import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

VLM_MODEL_ID = "/mnt/data/mritunjoyh/models/Qwen2-VL-7B-Instruct"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        print(f"Loading {VLM_MODEL_ID} (first call — this takes a few minutes)...")
        _processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        device = os.environ.get("VLM_DEVICE", "cuda:0")
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        _model.eval()
        print("VLM loaded.")


def vlm_chat(messages, max_new_tokens=512, temperature=0.0):
    """Call Qwen2-VL with OpenAI-format messages, return string response."""
    _load()
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(next(_model.parameters()).device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature > 0 and temperature != 1.0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    with torch.no_grad():
        ids = _model.generate(**inputs, **gen_kwargs)
    trimmed = ids[:, inputs.input_ids.shape[1]:]
    return _processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class _Choice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()


class _Response:
    def __init__(self, text):
        self.choices = [_Choice(text)]


class _Completions:
    def create(self, model, messages, max_tokens=512, temperature=0.0, **kwargs):
        text = vlm_chat(messages, max_new_tokens=max_tokens, temperature=temperature)
        return _Response(text)


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class LocalQwenClient:
    """Drop-in replacement for openai.OpenAI() that routes to local Qwen2-VL."""
    def __init__(self):
        self.chat = _Chat()
        self.api_key = "EMPTY"
