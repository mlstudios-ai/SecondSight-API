import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import sys
import re
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "api/src")))
from secondsight.model import ModelFactory, EnigmaAIVision

if __name__=="__main__":
    # instantiate once
    # model = ModelFactory.get_model(
    #     ModelFactory.ModelType.SCENE,
    #     ModelFactory.ModelName.ENIGMAAI)
    
    model = ModelFactory.get_model(
        ModelFactory.ModelType.SCENE,
        ModelFactory.ModelName.LLAVA)
    
    
    image_path = "/Users/jasper/Anna/Uni/UTS/MAI/Subjects/AIS/project/Datasets/inf/testimage.jpg"
    image = Image.open(image_path).convert("RGB")
    
    # generate a caption
    caption = model.predict(image, "Summarize the scene in this image:")
    print(caption)

