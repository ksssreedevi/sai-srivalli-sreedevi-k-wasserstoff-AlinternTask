import os
import sqlite3
import uuid
from PIL import Image

def save_objects(image, predictions, threshold=0.5, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = Image.open(image).convert("RGB")
    masks = predictions['masks']
    scores = predictions['scores']
    boxes = predictions['boxes']

    conn = sqlite3.connect('objects.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS objects (
            id TEXT PRIMARY KEY,
            image_id TEXT,
            bbox TEXT,
            score REAL,
            file_path TEXT,
            description TEXT
        )
    ''')

    image_id = str(uuid.uuid4())
    object_data = []

    for i, mask in enumerate(masks):
        if scores[i] > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            bbox = boxes[i].cpu().numpy().astype(int)
            object_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            object_id = str(uuid.uuid4())
            object_path = os.path.join(output_dir, f'{object_id}.png')
            object_image.save(object_path)

            c.execute('''
                INSERT INTO objects (id, image_id, bbox, score, file_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (object_id, image_id, str(bbox.tolist()), float(scores[i]), object_path))

            object_data.append({
                'id': object_id,
                'bbox': bbox.tolist(),
                'score': float(scores[i]),
                'file_path': object_path
            })

    conn.commit()
    conn.close()
    return object_data