
from PIL import Image
import matplotlib.pyplot as plt
import math
import glob 
import os

def show_generated_images(folder_path):
    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    
    num_images = len(image_files)
    
    if num_images == 0:
        print("No images found in the folder.")
        return

    num_rows = int(math.sqrt(num_images))
    num_cols = math.ceil(num_images / num_rows)
    
    plt.figure(figsize=(20, 20))
    
    for i, image_file in enumerate(image_files, start=1):
        img = Image.open(image_file)
        
        title = os.path.splitext(os.path.basename(image_file))[0].split("_")[-1]
        title = title.split('.')[0] # get the part between _ and .png
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(folder_path, title +'.png'))
    plt.show()


def show_indexed_images(folder_path, indices):
    plt.figure(figsize=(20, 20))
    for i, index in enumerate(unhealthy_indices, start=1):
        img_path = os.path.join(folder_path, f'cannabis-{index + 10}.png')
        img = Image.open(img_path)
        plt.subplot(6, 6, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Image index {index}')

    plt.tight_layout()
    plt.show()

