from skimage import io
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Define the color ranges for each color
color_ranges = {
    'Green': (np.array([20, 20, 10]), np.array([80, 220, 80])),
    'Yellow': (np.array([190, 190, 80]), np.array([255, 255, 130])),
    'Brown': (np.array([70, 40, 30]), np.array([180, 140, 105])),
    'Purple': (np.array([60, 40, 55]), np.array([200, 110, 200]))
}


# Convert each color range to RGB format
for color, (lower, upper) in color_ranges.items():
    lower_rgb = lower / 255.0
    upper_rgb = upper / 255.0
    color_ranges[color] = (lower_rgb, upper_rgb)

#We need to actually see the colors we are thresholding
#Therefore a colormap will help us visualize the colors

def colormap():
    # Create color map
    colors = []
    labels = []
    for color, (lower, upper) in color_ranges.items():
        colors.extend([lower, upper])
        labels.extend([f'{color} lower', f'{color} upper'])
    cmap = ListedColormap(colors)

    # Create figure with color bar
    fig, ax = plt.subplots(figsize=(10, 1))
    cb = plt.colorbar(ax.imshow(np.arange(len(colors)).reshape(1, len(colors)), cmap=cmap, aspect='auto'),
                      cax=ax, ticks=range(len(colors)))
    cb.ax.set_yticklabels(labels)
    plt.show()

# Define function process_images() to process a set of images by finding contours, 
# segmenting the images based on color ranges,
# and saving the segmented images and contours to an output folder.

# Define function process_images() to process a set of images by finding contours, 
# segmenting the images based on color ranges,
# and saving the segmented images and contours to an output folder.
def process_images(path, color_ranges=color_ranges):
    file_names = os.listdir(path)
    output_folder = 'output_images/'
    os.makedirs(output_folder, exist_ok=True)
    file_path = None  # Set file_path to None initially

    for file_name in file_names:
        if file_name.lower() == "hsv-kmeans-segmented.jpg" or file_name.lower() == "hsv-kmeans-segmented.jpeg" or file_name.lower() == "hsv-kmeans-segmented.png":
            print(f"Processing image '{file_name}'")
            file_path = os.path.join(path, file_name)
            print(f"File path: {file_path}")
            hsv = cv2.imread(file_path)

        if file_name.lower() == "hsv-kmeans-segmented-edges-.jpg" or file_name.lower() == "hsv-kmeans-segmented.jpeg" or file_name.lower() == "hsv-kmeans-segmented.png":
            print(f"Processing image '{file_name}'")
            file_path = os.path.join(path, file_name)
            print(f"File path: {file_path}")
            edges = cv2.imread(file_path, 0)

    if file_path and hsv is not None and edges is not None:  # Check if file_path has been assigned a value
        parent_folder = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        gray_filename = os.path.join(os.path.basename(file_path)[0] + f'_blur.png')
        gray = cv2.imread(os.path.join(parent_folder, gray_filename), 0)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        for color_name, (lower_range, upper_range) in color_ranges.items():
            mask = cv2.inRange(hsv, lower_range, upper_range)
            segmented = cv2.bitwise_and(hsv, hsv, mask=mask)
            cv2.imwrite(f'{output_folder}cannabis-{file_name}-{color_name}-Segmented.png', segmented)

            output_path = f'{output_folder}cannabis-{file_name}-Contour.png'
            cv2.imwrite(output_path, hsv)
            print(f"Image saved at: {output_path}")
            print("Image processing complete")
    else:
        print(f"Error: Image could not be read for file '{file_path}'")


#get_color_percentages(image, num_clusters) takes an image and the number of desired clusters as inputs, 
#and applies K-means clustering to the pixels of the image to obtain the dominant colors and their percentages. 
#he function returns the dominant colors in RGB values and their corresponding percentages.


def get_color_percentages(image, num_clusters):

    print("Image shape: ")
    print(type(image))
    #print(f"Image shape: {image.shape}")
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    pixels = np.reshape(-1, 3)

    # Apply K-means clustering to the pixels
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Count the number of pixels in each cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    # Calculate the percentage of pixels in each cluster
    percentages = [count / sum(pixel_counts.values()) * 100 for count in pixel_counts.values()]

    return kmeans.cluster_centers_.astype(int), percentages



def main(png):
    print(type(png))

    process_images(png)
    # Call the colormap function
    print("Colormap: ")
    colormap()
    # Call the get_color_percentages() function
    print("Color percentages: ")
    get_color_percentages(png,4)

