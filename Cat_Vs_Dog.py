import torch 
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import random
from tqdm.auto import tqdm
from helper_functions import accuracy_fn
from functions import train_step, test_step, eval_mode

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("D:/Python_projects/AI projects/Cat_Vs_Dog/")
image_path = data_path / "binary_animal_project"

#image_path.mkdir(parents= True, exist_ok= True)

#with zipfile.ZipFile("D:/Python_projects/archive.zip", "r") as zip_ref:
    #print("Extracting...")
    #zip_ref.extractall(image_path)
#print('Done')

def walk_through_dir(dirpath):
    """It will walk through every folder and tell you what's inside it"""
    for dir_path, dir_name, dir_file in os.walk(dirpath):
        print(f"There is {len(dir_name)} directiores and {len(dir_file)} images in {dir_path}")

train_path = image_path / "train" / "training_set"
test_path = image_path / "test" / "test_set"

image_path_list = list(train_path.glob("*/*.jpg"))

random.seed(42)

random_image = random.choice(image_path_list)

image_class = random_image.parent.stem

img = Image.open(random_image)

#plt.figure(figsize= (10, 7))
#plt.imshow(img)
#plt.title(image_class)
#plt.axis(False)

#image as numbers

image_as_num = np.asarray(img)

transformed_data = transforms.Compose([
    transforms.Resize(size= (64, 64)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image_converter(image_list, transform, n, seed= None):
    """Convert image into smaller one that is suitable for model to work on"""
    if seed:
        random.seed(seed)
        random_samples = random.sample(image_list, k= n)
        for image in random_samples:
            with Image.open(image) as I:
                fig, axis = plt.subplots(nrows= 1, ncols= 2)
                axis[0].imshow(I)
                axis[0].set_title(f"Original Size: {I.size}")
                axis[0].axis(False)

                transformed_images = transform(I).permute(1, 2, 0)
                axis[1].imshow(transformed_images)
                axis[1].set_title(f"Original Size: {transformed_images.shape}")
                axis[1].axis(False)
                plt.show()

#image_converter(image_list=image_path_list, transform=transformed_data, n= 5, seed= 42)

train_data = datasets.ImageFolder(
    root= train_path,
    transform= transformed_data,
    target_transform= None
)

test_data = datasets.ImageFolder(
    root= test_path,
    transform= transformed_data,
)

class_name = train_data.classes
class_idx = train_data.class_to_idx

#print(class_name, class_idx)  cats: 0 and dogs: 1

img, label = train_data[0][0], train_data[0][1]

img_resize = img.permute(1, 2, 0)

#plt.figure(figsize= (10, 7))
#plt.imshow(img_resize)
#plt.title(f"Resizable \nSize: {img_resize.shape} \nClass: {class_name[label]}")
#plt.axis(False)
#plt.show()

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset= train_data,
                              batch_size= BATCH_SIZE,
                              shuffle= True,
                              num_workers= 0)

test_dataloader = DataLoader(dataset= test_data,
                             batch_size= 32,
                             shuffle= False,
                             num_workers= 0)

class BinaryCNNModelV0(nn.Module):
    def __init__(self, input_num, output_num, hidden_units):
        super().__init__()
        self.CNN_1 = nn.Sequential(
            nn.Conv2d(in_channels= input_num, out_channels= hidden_units,
                      kernel_size= 3,
                      padding= 1,
                      stride= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units, out_channels= hidden_units,
                      kernel_size= 3,
                      padding= 1,
                      stride= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )

        self.CNN_2 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_units, out_channels= hidden_units,
                      kernel_size= 3,
                      padding= 1,
                      stride= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units, out_channels= hidden_units,
                      kernel_size= 3,
                      padding= 1,
                      stride= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )

        self.Linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units * 16 * 16, out_features= output_num)
        )

    def forward(self, x):
        x = self.CNN_1(x)
        x = self.CNN_2(x)
        x = self.Linear(x)
        return x

torch.manual_seed(42)

model_0 = BinaryCNNModelV0(input_num= 3, hidden_units= 10, output_num= 2).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params= model_0.parameters(),
                            lr= 0.001)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 5

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model_0, train_dataloader, optimizer, loss_fn, accuracy_fn, device)
    test_step(model_0, test_dataloader, optimizer, loss_fn, accuracy_fn, device)