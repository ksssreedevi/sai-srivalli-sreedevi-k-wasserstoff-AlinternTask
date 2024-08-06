import easyocr 
import numpy as np
from PIL import Image

def extract_text(object_data):
    reader = easyocr.Reader(['en'])
    for obj in object_data:
        image = Image.open(obj['file_path'])
        text = reader.readtext(np.array(image), detail=0)
        obj['extracted_text'] = ' '.join(text)
    return object_data

