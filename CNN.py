import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn, print_train_time
from tqdm.auto import tqdm
from timeit import default_timer as timer
from functions import train_step, test_step, eval_mode, make_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.FashionMNIST(
    root= "data", 
    train= True,
    download= True,
    transform= ToTensor(),
    target_transform= None
)

test_data = datasets.FashionMNIST(
    root= "data",
    train= False,
    download= True,
    transform= ToTensor(),
    target_transform= None
)

image, label = train_data[0]

name_photos = train_data.classes

torch.manual_seed(42)

fig = plt.figure(figsize=(10, 7))

for i in range(1, 4):
    random_idx = torch.randint(0, len(train_data), size= [1]).item()
    image, label = train_data[random_idx]

fig.add_subplot(4, 4, i)
plt.imshow(image.squeeze(), cmap= "gray")
plt.title(name_photos[label])
plt.axis(False)
#plt.show() To show all

BATCH_SIZE = 32

train_dataloader = DataLoader(
    dataset= train_data,
    batch_size= BATCH_SIZE,
    shuffle= True
)

test_dataloader = DataLoader(
    dataset= test_data,
    batch_size= BATCH_SIZE,
    shuffle= False
)

train_feature_batch, train_label_batch = next(iter(train_dataloader))

torch.manual_seed(42)

random_idx = torch.randint(0, len(train_feature_batch), size= [1]).item()
image, label = train_feature_batch[random_idx], train_label_batch[random_idx]

plt.imshow(image.squeeze(), cmap= "gray")
plt.title(name_photos[label])
plt.axis(False)

class CNNMODELV0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.CNN_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels= input_shape, out_channels= hidden_units,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units, out_channels= hidden_units,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)

        )
        self.CNN_layer_2 = nn.Sequential(
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units * 7 * 7, out_features= output_shape)
        )

    def forward(self, x):
        x = self.CNN_layer_1(x)
        x = self.CNN_layer_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)

model_0 = CNNMODELV0(input_shape= 1, hidden_units= 10,
                     output_shape= len(name_photos)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params= model_0.parameters(),
                            lr= 0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_timer = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model_0, train_dataloader, optimizer, loss_fn, accuracy_fn, device)
    test_step(model_0, test_dataloader, optimizer, loss_fn, accuracy_fn, device)

train_time_end_model_2 = timer()

total_train_time_on_model_2 = print_train_time(start= start_timer,
                                                end= train_time_end_model_2,
                                                device= device)

model_0_results = eval_mode(model_0, test_dataloader, optimizer, loss_fn, accuracy_fn, device)

import random

random.seed(42)

test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

plt.imshow(test_samples[0].squeeze(), cmap= "gray")
plt.title(name_photos[test_labels[0]])
pred_prob = make_predictions(model_0, test_samples, device)
pred_classes = pred_prob.argmax(dim= 1)

plt.figure(figsize=(10, 7))

for i, sample in enumerate(test_samples):
    plt.subplot(3, 3, i+1)

    plt.imshow(sample.squeeze(), cmap= "gray")

    pred_label = name_photos[pred_classes[i]]

    #Truth label
    truth_label = name_photos[test_labels[i]]

    #Create a title

    text_title = f"pred: {pred_label} , Truth: {truth_label}"

    if pred_label == truth_label:
        plt.title(text_title, c= "g")
    else:
        plt.title(text_title, c= "r")

    plt.axis(False)

plt.show()
