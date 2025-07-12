import torch
from transformers import AutoProcessor
from PIL import Image
import sys
import re
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "api/src")))
# from secondsight.model import ModelFactory, EnigmaAIVision

if __name__=="__main__":
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What’s shown in this image?"},
                ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This image shows a red stop sign."},]
        },
        {

            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in more details."},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Note that the template simply formats your prompt, you still have to tokenize it and obtain pixel values for your images
    print(text_prompt)

