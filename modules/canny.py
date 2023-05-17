
import os
import cv2
from showImages import show_generated_images

#This is a test on how to make a canny edge detector
# as the threshold settings are not immediately obvious
# therefore this can be run optionally, to figure out what threshold is best,
# and then the best threshold can be used in the main program
# however a lot of images are created.

def canny_test(preprocessed_folder):
    print(type(preprocessed_folder))
    high_threshold = 255
    low_threshold = high_threshold / 3

    output_folder = os.path.join(preprocessed_folder, 'canny')

    highThreshold_range = range(255//3, 256, 10)
    array_ab = [(low, high) for high in highThreshold_range for low in range(high//3, high, 10)]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(preprocessed_folder):
        image_path = os.path.join(preprocessed_folder, filename)

        # Check if the file is an image file before loading it
        if not image_path.endswith(".png"):
            continue

        # Load the image file
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image file was loaded successfully
        if image is None:
            print(f"Could not load image file: {image_path}")
            continue

        for a, b in array_ab:
            output_path = os.path.join(output_folder, f'{filename}-canny{a}_{b}.png')

            # Check if the output image already exists
            if os.path.exists(output_path):
                continue

            edges = cv2.Canny(image, a, b)
            cv2.imwrite(output_path, edges)

    show_generated_images(output_folder)


def main():
    print("CANNY - Canny edge threshold test")
    #
    # folderpath = input("Enter the folder path: ")
    # 
    folderpath = r"C:\\Users\\rober\\CC_09-1\\sh\\png\\preprocess\\"
    canny_test(folderpath)

main()