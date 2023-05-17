import sys
import os
import cv2
import numpy as np
from pathlib import Path
from skimage import io
from sklearn.cluster import KMeans

from colorProcess import color_ranges
from showImages import show_indexed_images

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)



# sys.path.append("..") # Add parent directory to Python path

def analyze_plant_health(input_folder, num_clusters=8, verbose=False):
    color_percentages = {}

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            file_path = os.path.join(input_folder, file_name)
            segmented_img = cv2.imread(file_path)

            if segmented_img is not None:
                # Reshape the image to be a list of pixels
                pixels = segmented_img.reshape(-1, 3)
                if verbose:
                    print("Image reshaped into a list of pixels")

                # Apply K-means clustering to the pixels
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(pixels)
                if verbose:
                    print("K-means clustering applied to pixels")

                # Count the number of pixels in each cluster
                unique, counts = np.unique(kmeans.labels_, return_counts=True)
                pixel_counts = dict(zip(unique, counts))
                if verbose:
                    print("Number of pixels in each cluster counted")

                # Calculate the percentage of pixels in each cluster
                percentages = [count / sum(pixel_counts.values()) * 100 for count in pixel_counts.values()]
                if verbose:
                    print("Percentage of pixels in each cluster calculated")

                unique_colors = kmeans.cluster_centers_.astype(int)

                for color, percentage in zip(unique_colors, percentages):
                    color_category = 'Other'
                    for color_name, (lower_range, upper_range) in color_ranges.items():
                        if (color >= lower_range).all() and (color <= upper_range).all():
                            color_category = color_name
                            break

                    color_percentages[color_category] = percentage

            else:
                print(f"Image could not be read for file '{file_path}'")

    return color_percentages


# Define a function to check the health of the plant
def is_plant_healthy(color_percentages, thresholds):
    problems = []

    if color_percentages['Yellow'] > thresholds['yellow']:
        problems.append('Yellowing leaves (possible nutrient deficiency)')

    if color_percentages['Brown'] > thresholds['brown']:
        problems.append('Browning leaves (possible nutrient burn or over-watering)')

    if color_percentages['Purple'] > thresholds['purple']:
        problems.append('Purpling leaves (possible phosphorus deficiency or cold stress)')

    return problems

def main():
    # Example usage
    input_folder = input("Enter input folder path: ")
    # kmeans_folder = os.path.join(kmeans_folder, 'png')
    # os.makedirs(kmeans_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"The specified path '{input_folder}' does not exist.")
    else:
        file_names = os.listdir(input_folder)
        if not file_names:
            print(f"The specified path '{input_folder}' does not contain any files.")
        else:
            print(f"The specified path '{input_folder}' contains the following files:")
            for file_name in file_names:
                print(file_name)





    image_color_percentages = analyze_plant_health(input_folder, num_clusters=4,verbose=True)

    color_names = [color_info[2] for color_info in color_ranges]
    color_names.append('Other')

    array_data = np.zeros((len(image_color_percentages), len(color_names)))

    for i, color_percentages in enumerate(image_color_percentages):
        for j, color_name in enumerate(color_names):
            array_data[i, j] = color_percentages.get(color_name, 0)

    print("Color percentages array:")
    print(array_data)


    # Define threshold values for color ranges (in percentages)
    thresholds = {
        'yellow': 10,
        'brown': 5,
        'purple': 5
    }


    # Initialize arrays for healthy and unhealthy indices
    healthy_indices = []
    unhealthy_indices = []

    # Iterate through the images and check if the plant is healthy or not
    for idx, color_percentages in enumerate(array_data):
        problems = is_plant_healthy(color_percentages, thresholds)
        if problems:
            unhealthy_indices.append(idx)
            print(f"Image index {idx} has the following problems:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            healthy_indices.append(idx)
            print(f"Image index {idx} appears to be healthy.")

    # try:
    #     # Print the indices of healthy and unhealthy plants
    #     print("Healthy indices:", healthy_indices)
    #     show_indexed_images(input_folder, healthy_indices)

    #     # Show the unhealthy images
    #     print("Unhealthy indices:", unhealthy_indices)
    #     show_indexed_images(input_folder, unhealthy_indices)
    # except NameError as e:
    #     print("Error:", e)
    #     print("One or more indices are not defined.")

if __name__ == "__main__":
    main()