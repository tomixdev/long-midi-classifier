import numpy as np
import step3_util

import torchvision.transforms as transforms

import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split

import os
import step3_params

# Settings ----------------------------------------------------------------------------------------
step3_util.set_plt_config()
step3_util.set_np_config()

device = step3_util.set_and_get_device()
step3_util.torch_seed(step3_params.RAND_SEED) 

# Prepare Dataset ---------------------------------------------------------------------------------
# Define transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=step3_params.interpolation_method),
    transforms.ToTensor()
])

test_transform = copy.deepcopy(train_transform)

# Load the dataset from the "dog" and "cat" folders
full_dataset = datasets.ImageFolder(root=step3_params.DATA_DIR, transform=train_transform)

# Split the dataset into train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Define Data Loader
batch_size = step3_params.batch_size

# Create data loaders for the train and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Prepare Model -----------------------------------------------------------------------------------
model = step3_params.pretraine_model.to(device)

# 学習の設定
n_epochs = step3_params.n_epochs # どうも10くらいをすぎると過学習が起こるみたいだ。
lr = step3_params.lr

# get directory names one level under DATA_DIR
label_list = [d.name for d in os.scandir(step3_params.DATA_DIR) if d.is_dir()]
# sort list
label_list.sort()

# Freeze all layers except the last fully connected layer
for param in model.parameters():
    param.requires_grad = False

model.fc.require_grad = True

# Replace the last fully connected layer with a new one for grayscale image classification
num_classes = len(label_list)
model.fc = torch.nn.Linear(512, num_classes, device=device)

# Define the loss function and optimizer
criterion = step3_params.criterion.to(device)
# パラメータ修正の対象を最終のfcに限定する。
optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=step3_params.optim_sgd_momentum)

# Train Model -------------------------------------------------------------------------------------
# history 初期化
history = np.zeros((0, 5))

history = step3_util.fit(
    net=model, optimizer=optimizer, criterion=criterion, num_epochs=n_epochs, train_loader=train_loader, test_loader=test_loader, device=device, history=history
)

# Visualize Model ---------------------------------------------------------------------------------
step3_util.evaluate_history(history)
step3_util.show_images_labels(test_loader, label_list, net=model, device=device, figsize=(120, 15))

