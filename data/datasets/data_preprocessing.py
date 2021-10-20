""" load data
    Dataset: access and preprocess data from a file or data source
    Sampler: sample data from a dataset in order to create batches
    DataLoader: iterate over a set of batches
    torchvision
    torchtext
    Torchaudio
""" 

""" 
from pickle import TRUE
import torch
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import T
train_data = CIFAR10(root="./train/",
                    train=True,
                    download=True)
print(train_data)
print(len(train_data))
print(train_data.data.shape)
print(train_data.classes)
print(train_data.targets) # A list of data labels
print(train_data.class_to_idx)

# detailed about the data
print(type(train_data[0]))
print(len(train_data[0]))
data, label = train_data[0]
print(data) # <PIL.Image.Image image mode=RGB size=32x32 at 0x7F4455B23610>
print(label)
print(train_data.classes[label])

test_data = CIFAR10(root="./test/",
                    train=False,
                    download=True)

# PIL(use Pillow library to store iamge pixel values in the format of heightxwidthxchannels)
""" 
#-------------------------------------------------------------------------------------------------------------------------
""" apply transforms
value normalize
augmentation
convert to tensor
"""


"""from pickle import TRUE
import torch
from torch._C import _multiprocessing_init
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

# train data
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))])

train_data = CIFAR10(root="./train/",
                    train=True,
                    download=True,
                    transform=train_transforms)


# test data
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
      (0.2023, 0.1994, 0.2010))])

test_data = CIFAR10(
    root="./test/",
    train=False,
    download=True,
    transform=test_transforms)"""





#---------------------------------------------------------------------------------------------------------
"""
batch: train model need pass in small batches of data at each iteration
        model development: takes advantage of the parallel nature of GPU to accelerate training
        torch.utils.data.DataLoader"""

trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=16,
    shuffle=True
)

data_batch, label_batch = next(iter(trainloader)) # iter to cast the trainloader, next to iterate over the data one more time 

testloader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=16,
                    sampler=None,
                    batch_sampler=None,
                    num_workers=0, # increase the number of CPU processes that generate batches in parallel
                    collate_fn=None,
                    pin_memnory=False,
                    drop_last=False,
                    timeout=0，
                    worker_init_fn=None,
                    _multiprocessing_context=None,
                    generator=0,
                    shuffle=False)

#----------------------------------------------------------------------------------------------------------
"""
torch.utils.data: create own data and dataloader classes 
it consists of Dataset, Sampler, DataLoader"""

# create your own dataset class 
        # map-style dataset
        # torch.utils.data.Dataset (getitem((),len() functions))

    # iterable-style dataset
        # torch.utils.data.IterableDataset 
        #reading data from a database or remote server, data generated in real tiem 
"""
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        """