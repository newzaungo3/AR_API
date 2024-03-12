from typing import Union
import requests
import cv2
import urllib.request 
from fastapi import FastAPI
import os
from PIL import Image
import open_clip
import torch
from config import * 
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

api_endpoint = "https://threed-model-management.onrender.com/model-management-system/internal/ml/classification/store"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJCbnVIRzQxbXpkdE1ZMjhuM3JSeCIsIm9yZ2FuaXphdGlvbklkIjoieU9oOWhBYmQ0RmZ1OTRtUHBLcjgiLCJpYXQiOjE3MDEwNjE5NTl9.cZ2xkoMJzvyaMMUYYm15XiG4xA9YmFS8fuZJBpf6d4Y"
}
label_dict = dict()
# Make a GET request to the API
response = requests.post(api_endpoint,headers=headers)
base_path = './dataset/api_dataset/'
json_path = './dataset/json/'

model_name = 'coca_ViT-L-14' #'ViT-L-14-CLIPA' ,coca_ViT-L-14
pretrained = 'mscoco_finetuned_laion2b_s13b_b90k' #'datacomp1b' ,mscoco_finetuned_laion2b_s13b_b90k
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                             device = device,
                                                             pretrained=pretrained)
query_classes = []
q_batch =  []
count = 0
#print(preprocess)
# Define transforms for each augmentation
preprocess_original = preprocess  # Your original preprocess function for clear RGB images
preprocess_grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    preprocess
])
preprocess_blur = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    preprocess
])
preprocess_zoom_out = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    # Step 1: Resize the image to make it smaller than the model's expected input dimensions
    transforms.Resize(180),  # Assuming the original input size is 256x256, adjust this value as needed
    # Step 2: Pad the resized image to match the model's expected input dimensions
    # The padding mode can be changed as needed (e.g., 'constant', 'edge', 'reflect')
    lambda img: F.pad(img, padding=(38, 38, 38, 38), padding_mode='constant', fill=0),
    # Step 3: Apply your model-specific preprocessing (e.g., normalization)
    preprocess
])
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/download/")
def read_item():
    base_path = '/dataset/api_dataset/'
    
    if response.status_code == 200:
        # Parse the response as JSON (assuming the API returns JSON data)
        data = response.json()

        # Now you can work with the data
        print(len(data))
        for i in range(len(data)):
            #print(i)
            #print(type(data[i]))
            print(data[i])
            print(type(data[i]['markers']))
            print(f"marker: {data[i]['markers']}")
            marker_list = data[i]['markers']
            for j in range(len(marker_list)):
                print(marker_list[j]['s3Url'])
                fullfilename = os.path.join(base_path, marker_list[j]['modelId'])
                url = marker_list[j]['s3Url']
                urllib.request.urlretrieve(url, f"{marker_list[j]['s3s3FileName']}") 
                # Accessing the content of each dictionary
    else:
        # If the request was not successful, print an error message
        print(f"Error: {response.status_code}")
        print(response.text)
    
    
    return {"Start downloading"}