import time
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline

start_time = time.time()

weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()
model_load_time = time.time()
print(f"Model loading time: {model_load_time - start_time} seconds")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

summarizer = pipeline("summarization")

def segment_image(image_path):
    image_processing_time = time.time()
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    with torch.no_grad():
        predictions = model([image_tensor])
    print(f"Image processing time: {time.time() - image_processing_time} seconds")
    return predictions[0]



