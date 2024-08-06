import cv2
import os

def visualize_segments(image_path, predictions, output_dir='output', threshold=0.5):
    image = cv2.imread(image_path)
    masks = predictions['masks']
    scores = predictions['scores']

    for i, mask in enumerate(masks):
        if scores[i] > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    visualized_image_path = os.path.join(output_dir, 'visualized_image.png')
    cv2.imwrite(visualized_image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))