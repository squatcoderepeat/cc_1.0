#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import sys
import importlib

# Add the path to the 'modules' directory to the system path
print("Adding modules directory to system path")
sys.path.append(r"C:\Users\rober\CC_09-1\modules")

# list all the files in the modules directory

# verbosity if needed for troubleshooting
# print("Listing files in modules directory")
os.listdir(r"C:\Users\rober\CC_09-1\modules")


# Load the 'modules' package


from preProcess import main as preprocess_main
from colorProcess import main as colorprocess_main
from analyze import main as analyze_main   
from canny import canny_test


import modules

for module_file in os.listdir(modules.__path__[0]):
    print("Importing", module_file)
    if module_file.endswith('.py') and module_file != '__init__.py' :        
        module_name = module_file[:-3]  # Remove the '.py' extension
        module = importlib.import_module(f'{module_name}')
        
        # Import all functions and classes from the module into the global namespace
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr):
                globals()[attr_name] = attr



def main():
    # 1. Preprocess the images
    # This will execute all our code in Preprocess.py
    # which takes in a RAW image, converts it into PNG
    # then takes the PNG and preproccesses it and saves it 
    # eventually as a grayscale

    print("MAIN - Preprocessing images")
    preprocess_plant = preprocess_main()

    # 2. Run Canny test
    # This is a test on how to make a canny edge detector
    # as the threshold settings are not immediately obvious

    print("CANNY - Canny edge threshold test")
    canny_test(preprocess_plant)


    # 3.Run K-means
    # This will execute all our code in K-means.py
    # which will run K-means clustering on the flattened image
    # and then will reshape the cluster centers back into the original image shape
    # and then will save the segmented image to the output folder
    
    #print("K-Means - Preprocessing images")
    
    
    #kmeans_output = kmeans(preprocess_plant)     
    
    #print("K-Means - Done")

    # 4.
    # Color process the images
    # This will execute all our code in Colorprocess.py
    # which will show us the colors available in the thresholds
    # and then will show us the images with the colors segmented
   #print("Colorprocessing images")

    #print(type(kmeans_output))

    #colorprocess_main(kmeans_output)

    # 5. Analyze the images
    # This will execute all our code in Analyze.py
    # which will analyze the images and show us the percentages of the colors
    # and then will show us the images with the colors segmented
    #print("Analyze images")
    #analyze_main()
    
    #6. Show the images
    
    #show_indexed_images(folder_path, unhealthy_indices)    #Run the process_images function from the analyze module

    #process_images()

main()  