
def extract_features_plants(preprocessed_folder, save_folder):

    preprocessed_files = glob.glob(os.path.join(preprocessed_folder,"*.*"))
    features_array = None
    print(f"Number of files found: {len(preprocessed_files)}")
    for i, preprocessed_file in enumerate(preprocessed_files):
        img = cv2.imread(preprocessed_file)
        print(f"Processing file: {preprocessed_file}")
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        gray_file = os.path.join(save_folder, os.path.basename(preprocessed_file))
        cv2.imwrite(gray_file, gray_img)
        print(f"Grayscale image saved: {gray_file}")
        # Extract the histogram of oriented gradients (HOG) feature
        hog_feature = feature.hog(gray_img)

        # Gaussian blur
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        # Save the blurred image
        blur_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_blur.png')
        cv2.imwrite(blur_file, blur)

        edges = cv2.Canny(gray_img, 48, 85)

        # Save the edges image
        edges_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_edges.png')
        cv2.imwrite(edges_file, edges)
        

        # Compute the color histogram
        color_hist = color.rgb2gray(img).flatten()

        # Convert the image to HSV color space to identify the location of different colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Save the HSV image
        hsv_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_hsv.png')
        cv2.imwrite(hsv_file, hsv)
        print(f"HSV image saved: {hsv_file}")
        # Combine the features
        combined_features = np.concatenate((hog_feature, color_hist))
        num_features = len(hog_feature) + len(color_hist)

        if features_array is None:
            features_array = np.empty((len(preprocessed_files), num_features))

        features_array[i] = combined_features

    return features_array


# This is a relic from a function that was made to write the 
# pictures of the plant labels into the subfolders for each of the images,
# so for example 00023.png would be the first image of a label, and it may be
# "purple african 3" so it would be written to the folder "purple african 3"

def extract_features_labels(preprocessed_folder, save_folder):
    preprocessed_files = glob.glob(os.path.join(preprocessed_folder, "*.png"))
    features_array = None

    for i, preprocessed_file in enumerate(preprocessed_files):
        img = cv2.imread(preprocessed_file)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        gray_file = os.path.join(save_folder, os.path.basename(preprocessed_file))
        cv2.imwrite(gray_file, gray_img)

        # Extract the histogram of oriented gradients (HOG) feature
        hog_feature = feature.hog(gray_img)

        # Gaussian blur
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        # Save the blurred image
        blur_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_blur.png')
        cv2.imwrite(blur_file, blur)

        # Canny edge detection
        edges = cv2.Canny(gray_img, 78, 205)

        # Save the edges image
        edges_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_edges.png')
        cv2.imwrite(edges_file, edges)

        # Compute the color histogram
        color_hist = color.rgb2gray(img).flatten()

        # Convert the image to HSV color space to identify the location of different colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Save the HSV image
        hsv_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_hsv.png')
        cv2.imwrite(hsv_file, hsv)

        # Combine the features
        combined_features = np.concatenate((hog_feature, color_hist))
        num_features = len(hog_feature) + len(color_hist)

        if features_array is None:
            features_array = np.empty((len(preprocessed_files), num_features))

        features_array[i] = combined_features

    return features_array
