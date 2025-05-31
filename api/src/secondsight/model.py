"""
Base abstract class for object detection. 

Author: Anna Huang
Date: 16 May 2025
License: MIT License
Copyright (c) 2024 Anna Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Union, Optional
import io
from pathlib import Path
from enum import Enum
from PIL import Image
from ultralytics import YOLO
import torch
import gc
import re
from transformers import (
    AutoProcessor,
    ViTFeatureExtractor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration
)
from secondsight import util

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class ModelLoadError(ModelError):
    """Exception raised when model fails to load."""
    pass

class PredictionError(ModelError):
    """Exception raised when model prediction fails."""
    pass

class BaseModel(ABC):
    """
    The abstract base class. 
    
    Args:
        ABC (_type_): Python abstract class.
    """
    _device = util.get_device_name()
    
    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        
    @abstractmethod    
    def predict(self, image: Union[Image.Image, bytes], **kwargs) -> Any:
        """Predict using the model.
        
        Args:
            image: Either a PIL Image or image bytes
            **kwargs: Additional arguments for prediction
            
        Returns:
            Any: Model prediction result
            
        Raises:
            PredictionError: If prediction fails
        """
        pass
    
    def finalize(self):
        """Clean up model resources in a platform-agnostic way."""
        try:
            # Move model to CPU before deletion to ensure proper cleanup
            if hasattr(self, '_model'):
                if hasattr(self._model, 'to'):
                    self._model.to('cpu')
                # Clear device-specific caches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                del self._model

            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error during model cleanup: {str(e)}")

class DetectionModel(BaseModel):
    def __init__(self, model_name: str, classes: List[str], conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        super().__init__(model_name)
        
        self.__classes = classes
        self.__conf = conf
        self.__iou = iou
        self.__imgsz = imgsz
        try:
            self.__model = YOLO(model_name)
        except Exception as e:
            raise ModelLoadError(f"Failed to load detection model: {str(e)}")
           
    def predict(self, image: Union[Image.Image, bytes], **kwargs) -> Any:
        try:
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            return self.__model.predict(
                image,
                conf=self.__conf,
                iou=self.__iou,
                imgsz=self.__imgsz,
                classes=self.__classes,
                **kwargs
            )
        except Exception as e:
            raise PredictionError(f"Detection prediction failed: {str(e)}")

class SceneModel(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)         
        
    def finalize(self):
        """Clean up model resources in a platform-agnostic way."""
        try:
            super().finalize()

            # Clean up processor
            if hasattr(self, '_processor'):
                del self._processor

            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error during model cleanup: {str(e)}")
            
    @abstractmethod 
    def predict(self, image: Union[Image.Image, bytes], prompt: str, **kwargs) -> str:
        """Predict using the scene model.
        
        Args:
            image: Either a PIL Image or image bytes
            prompt: Text prompt for the model
            **kwargs: Additional arguments for prediction
            
        Returns:
            str: Model prediction result
            
        Raises:
            PredictionError: If prediction fails
        """
        pass
      
class AyaVision(SceneModel):
    def __init__(self, model_name: str):        
        super().__init__(model_name)   
        try:
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModelForImageTextToText.from_pretrained(model_name).to(self._device)
        except Exception as e:
            raise ModelLoadError(f"Failed to load AyaVision model: {str(e)}")
        
    def predict(self, image: Union[Image.Image, bytes], prompt: str, **kwargs) -> str:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
    
        # Use chat template if required by your model
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._device)
        output_ids = self._model.generate(**inputs)
        return self._processor.decode(output_ids[0], skip_special_tokens=True)

    
class LlavaVision(SceneModel):
    def __init__(self, model_name: str):        
        super().__init__(model_name)   
        try:            
            self._model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(self._device)
            self._processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            raise ModelLoadError(f"Failed to load LlavaVision model: {str(e)}")
        
    def predict(self, image: Union[Image.Image, bytes], prompt: str, **kwargs) -> str:
        try:
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._device)
            output_ids = self._model.generate(**inputs)
            return self._processor.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            raise PredictionError(f"LlavaVision prediction failed: {str(e)}")

class EnigmaAIVision(SceneModel):    
    BASE_DIR = Path(__file__).parent.parent.parent
    
    def __init__(self, model_name: str):
        super().__init__(self.BASE_DIR / model_name)   
    
        try:
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self._device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.decoder.resize_token_embeddings(len(tokenizer))        
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            
            self._model = model
            self.tokenizer = tokenizer
            self.feature_extractor = feature_extractor
        except Exception as e:
            raise ModelLoadError(f"Failed to load EnigmaAIVision model: {str(e)}")
        
    def predict(self, image: Union[Image.Image, bytes], prompt: str, **kwargs) -> str:
        self._model.eval()
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self._device)
        kwargs["pixel_values"] = pixel_values

        # tokenize prompt
        prompt_inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_ids = prompt_inputs.input_ids.to(self._device)
        prompt_len = prompt_ids.size(-1)
        
        default_kwargs = {
            "pixel_values": pixel_values,
            "decoder_input_ids": prompt_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": 50,
            "max_new_tokens": 50,
            "min_length": 10,
            "num_beams": 3,
            "length_penalty": 0.7,
            "no_repeat_ngram_size": 3,
        }
        
        kwargs = {**default_kwargs, **kwargs}
        
        with torch.no_grad():
            output_ids = self._model.generate(**kwargs)
            
        # slice away the prompt tokens
        gen_ids = output_ids[0][prompt_len:]
        
        # decode and strip any leading junk
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        cleaned = re.sub(r'^[^A-Za-z0-9]+', '', raw).strip()
        if "." in cleaned:
            cleaned = cleaned[: cleaned.rfind(".") + 1 ]
        return cleaned
            
class ModelFactory:
    
    class ModelType(Enum):
        DETECTION = 0
        SCENE = 1
        
    class ModelName(Enum):
        LLAVA = "llava-hf/llava-1.5-7b-hf"      # from Hugging Face
        AYA = "CohereLabs/aya-vision-8b"        # from Hugging Face
        ENIGMAAI = "api/models/SceneModel"      # custom trained model       
    
    @staticmethod
    def get_model(type: ModelType, name: ModelName) -> BaseModel:
        """Get a model instance based on type and name.
        
        Args:
            type: Type of model to get
            name: Name of the specific model
            
        Returns:
            BaseModel: Instance of the requested model
            
        Raises:
            ValueError: If invalid model type or name is provided
        """
        try:
            if type == ModelFactory.ModelType.DETECTION:
                raise NotImplementedError("Detection models not yet implemented")
            elif type == ModelFactory.ModelType.SCENE:
                if name == ModelFactory.ModelName.AYA:
                    return AyaVision(name.value)
                elif name == ModelFactory.ModelName.LLAVA:
                    return LlavaVision(name.value)
                elif name == ModelFactory.ModelName.ENIGMAAI:
                    return EnigmaAIVision(name.value)
                else:
                    raise ValueError(f"Invalid scene model name: {name}")
            else:
                raise ValueError(f"Invalid model type: {type}")
        except Exception as e:
            raise ModelError(f"Failed to create model: {str(e)}")