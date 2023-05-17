import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def kmeans(input_folder, num_clusters=8, verbose=False):

    print("Running KMeans clustering on images in the input folder...")

    # Define the source and destination paths
    folder_path_1 = Path(input_folder)
    
    output_folder = folder_path_1 / "kmeans_output"
    masked_folder = folder_path_1 / "masked_output"

    # Create the output folder if it doesn't exist
    try:
        output_folder.mkdir(parents=True)
    except FileExistsError:
        pass
        
    try:
        masked_folder.mkdir(parents=True)
    except FileExistsError:
        pass

    print("Folders created")

    # Check if the input folder is empty
    if not list(folder_path_1.glob("*")):
        print("Error: Input folder is empty.")
        return

    
    # Get the total number of files in the input folder
    total_files = len(list(folder_path_1.glob("*.png")))


    # Loop over each file in the source folder
    for i, filename in enumerate(folder_path_1.iterdir()):
        if filename.name.endswith('_hsv.png'):
            if verbose:
                print(f"Processing {filename.name} ({i+1}/{total_files})")
                
            # Load the preprocessed image
            image = cv2.imread(str(filename))
            if verbose:
                print(f"Image read")
            # Flatten the image into a 2D array of pixels
            pixels = np.reshape(image, (-1, 3))
            if verbose:
                print(f"Image reshaped into 2d array")
            # Run KMeans clustering on the flattened image
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(pixels)
            if verbose:
                print(f"Image clustered")
            # Reshape the cluster centers back into the original image shape
            cluster_centers = kmeans.cluster_centers_.astype('uint8')
            segmented_image = cluster_centers[kmeans.labels_]
            segmented_image = np.reshape(segmented_image, image.shape)
            if verbose:
                print(f"Image segmented")
            # Save the segmented image to the output folder
            output_path_1 = output_folder / f"{filename.stem}-KMeans-Segmented.png"
            cv2.imwrite(str(output_path), segmented_image)
            if verbose:
                print(f"Image saved")
            edges = cv2.Canny(segmented_image, 100, 200)
            output_path = output_folder / f"{filename.stem}-KMeans-Segmented-edges.png"
            cv2.imwrite(str(output_path), edges)
            if verbose:
                print(f"Image edges saved")

            #create a binary mask from the edges
            ret,thresh = cv2.threshold(edges,127,255,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)
            cnt = contours[0]
            mask = np.zeros_like(segmented_image)
            cv2.drawContours(mask, [cnt], 0, (100,177,255), -1)
            output_path = masked_folder / f"{filename.stem}-KMeans-Segmented-mask.png"
            cv2.imwrite(str(output_path), mask)
            if verbose:
                print(f"Image mask saved")
            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(image, mask)
            output_path = masked_folder / f"{filename.stem}-KMeans-Segmented-masked.png"
            cv2.imwrite(str(output_path), masked_image)
            if verbose:
                print(f"Image masked saved")
            # Save the masked image to the output folder
    return output_path_1



if __name__ == "__main__":
    input_folder = input("Enter input folder path: ")
    kmeans(input_folder,verbose=True)
    