import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import bayer  

def read_image(path):
    """Простая функция загрузки, чтобы не зависеть от common.py"""
    if not path.endswith('.png'):
        path += '.png'
    
    try:
        raw_img = np.array(Image.open(path))
        if raw_img.ndim == 3:
             raw_img = raw_img.mean(axis=2).astype(np.uint8)
    except FileNotFoundError:
        print(f"Ошибка: Файл {path} не найден!")
        return None, None

    dirname, filename = os.path.split(path)
    gt_path = os.path.join(dirname, "gt_" + filename)
    
    gt_img = None
    if os.path.exists(gt_path):
        gt_img = np.array(Image.open(gt_path))
    
    return raw_img, gt_img

def plot_bayer_comparison(image_path):
    raw_img, gt_img = read_image(image_path)
    if raw_img is None:
        return

    print(f"Обработка изображения: {image_path}")
    
    result_img = bayer.get_colored_img(raw_img.astype(np.uint8))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    h, w = raw_img.shape
    cy, cx = h // 2, w // 2
    if h > 20 and w > 20:
        zoom_raw = raw_img[cy:cy+20, cx:cx+20]
    else:
        zoom_raw = raw_img
    
    axes[0].imshow(zoom_raw, cmap='gray')
    axes[0].set_title("Input: Bayer Pattern (Zoom 20x20)", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(result_img.astype(np.uint8))
    axes[1].set_title("Result: Demosaicing", fontsize=14)
    axes[1].axis('off')

    if gt_img is not None:
        axes[2].imshow(gt_img.astype(np.uint8))
        axes[2].set_title("Ground Truth (Original)", fontsize=14)
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, "GT not found", ha='center')
    
    plt.tight_layout()
    output_name = "bayer_viz.png"
    plt.savefig(output_name, dpi=150)
    plt.show()

if __name__ == "__main__":
    test_file = "tests/08_unittest_improved_img_fast_input/01" 
    plot_bayer_comparison(test_file)