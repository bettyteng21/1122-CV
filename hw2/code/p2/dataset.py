import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloader(dataset_dir, batch_size=1, split='test'):
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            ##### TODO: Data Augmentation Begin #####
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((64,64), padding=4, padding_mode='symmetric'), 
            ##### TODO: Data Augmentation End #####
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif split == 'unlabel':
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            ##### TODO: Data Augmentation Begin #####
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((32,32), padding=4, padding_mode='symmetric'), 
            ##### TODO: Data Augmentation End #####
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    dataset = CIFAR10Dataset(dataset_dir, split=split, transform=transform)

    if dataset[0] is None:
        raise NotImplementedError('No data found, check dataset.py and implement __getitem__() in CIFAR10Dataset class!')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0, pin_memory=True, drop_last=(split=='train'))

    return dataloader

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None):
        super(CIFAR10Dataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)
        
        self.image_names = json_data['filenames']
        if self.split != 'test':
            self.labels = json_data['labels']

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        ########################################################
        # TODO:                                                #
        # Define the CIFAR10Dataset class:                     #
        #   1. use Image.open() to load image according to the # 
        #      self.image_names                                #
        #   2. apply transform on image                        #
        #   3. if not test set, return image and label with    #
        #      type "long tensor"                              #
        #   4. else return image only                          #
        #                                                      #
        # NOTE:                                                #
        # You will not have labels if it's test set            #
        ########################################################

        img_path = os.path.join(self.dataset_dir, str(self.image_names[index]))
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img_tranformed = self.transform(img)
        
        if self.split != 'test' and self.split != 'unlabel':
            label = torch.tensor(self.labels[index], dtype=torch.long)
            return {
                'images': img_tranformed, 
                'labels': label
            }
        else:
            return {
                'images': img_tranformed
            }
