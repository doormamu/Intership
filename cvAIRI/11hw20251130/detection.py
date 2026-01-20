import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from skimage.transform import resize

from config import VOC_CLASSES, bbox_util, model
from utils import get_color


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)


def image2tensor(image):
    # Write code here
    image = Image.fromarray(image[..., ::-1]).resize((300, 300), Image.BILINEAR)
    image = np.array(image, dtype=np.float32)
    #image = ...  # convert RGB to BGR
    image = image - IMAGENET_MEAN 
    image = image.transpose([2, 0, 1])  # torch works with CxHxW images
    tensor = torch.tensor(image.copy()).unsqueeze(0).float()
    # tensor.shape == (1, channels, height, width)
    return tensor


@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    # Write code here
    # First, convert the input image to tensor 
    h, w= frame.shape[:2]              
    input_tensor = image2tensor(frame)
    
    if next(model.parameters()).is_cuda:
        input_tensor = input_tensor.cuda()
    
    # Then use model(input_tensor),
    # convert output to numpy
    # and bbox_util.detection_out
    results = bbox_util.detection_out(model(input_tensor), confidence_threshold=min_confidence)
    if not results or len(results[0]) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    print(results)

    # Select detections with confidence > min_confidence
    # hint: see confidence_threshold argument of bbox_util.detection_out
    


    # If label set is known, use it
    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indices = [
            index
            for index, label in enumerate(result_labels)
            if VOC_CLASSES[label - 1] in labels
        ]
        results = results[indices]

    results = results[0]
    mask = results[:, 1] > min_confidence 
    results = results[mask]
    # Remove confidence column from result
    label = results[:, 0]
    coords = results[:, 2:]
    results = np.column_stack([label, coords])

    # Resize detection coords to the original image shape.
    #print(results.shape)

    results[:, 1::2] *= w
    results[:, 2::2] *= h
    results = np.round(results).astype(np.int32)
    print(results)

    # Return result
    return detection_cast(results)


def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """
    frame = frame.copy()

    img = Image.fromarray(frame[..., ::-1])
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        label_id = int(det[0]) - 1
        xmin, ymin, xmax, ymax = map(int, det[1:5])

        color_bgr = get_color(label_id)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color_rgb, width=3)

        class_name = VOC_CLASSES[label_id] if 0 <= label_id < len(VOC_CLASSES) else f"class_{label_id}"
        text = class_name

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle([xmin, ymin - text_h - 6, xmin + text_w + 10, ymin],
                       fill=color_rgb)

        draw.text((xmin + 5, ymin - text_h - 3), text,
                  fill=(255, 255, 255), font=font)

    result = np.array(img)[..., ::-1].copy()
    return result



def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, "data", "test.png"))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == "__main__":
    main()
