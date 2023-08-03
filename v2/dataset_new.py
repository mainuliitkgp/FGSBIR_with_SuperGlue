import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
import cv2
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        self.root_dir = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name)
        coordinate_path = os.path.join(self.root_dir , hp.dataset_name + '_Coordinate')
        with open(coordinate_path, 'rb') as fp:
            Coordinate = pickle.load(fp) # {'sketch_name': [(x, y)]}

        photo_path = os.path.join(self.root_dir, 'photo')
        target_path = os.path.join(self.root_dir, 'error_files')
        image_error_path = os.path.join(target_path, 'image_error_filelist.pkl')

        with open(image_error_path, 'rb') as fp:
            image_error_filelist = pickle.load(fp)

        Coordinate_mod = {}
        for item in Coordinate.keys():
            positive_sample = '_'.join(item.split('/')[-1].split('_')[:-1])
            postive_sample_name = positive_sample+'.png'
            if postive_sample_name not in image_error_filelist:
                Coordinate_mod[item] = Coordinate[item]
        
        self.Coordinate = Coordinate_mod

        self.Train_Sketch = [x for x in self.Coordinate if 'train' in x] 
        self.Test_Sketch = [x for x in self.Coordinate if 'test' in x]

        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')

    def __getitem__(self, item):
        sample  = {}
        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            positive_sample = '_'.join(self.Train_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')

            possible_list = list(range(len(self.Train_Sketch)))
            possible_list.remove(item)
            negative_item = possible_list[randint(0, len(possible_list) - 1)]
            negative_sample = '_'.join(self.Train_Sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = self.train_transform(Image.fromarray(sketch_img).convert('RGB')).mean(0).unsqueeze(0).to(device)

            positive_img = self.train_transform(Image.open(positive_path).convert('RGB')).mean(0).unsqueeze(0).to(device)
            negative_img = self.train_transform(Image.open(negative_path).convert('RGB')).mean(0).unsqueeze(0).to(device)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'negative_img': negative_img, 'negative_path': negative_sample
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = self.test_transform(Image.fromarray(sketch_img).convert('RGB')).mean(0).unsqueeze(0).to(device)

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = self.test_transform(Image.open(positive_path).convert('RGB')).mean(0).unsqueeze(0).to(device)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'Coordinate':vector_x,
                      'positive_img': positive_img, 'positive_path': positive_sample}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def get_dataloader(hp):

    dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True)#,
    #                                     num_workers=int(hp.nThreads))

    dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False)#,
    #                                     num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test

def get_transform(type):
    transform_list = []
    transform_list.extend([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
    return transforms.Compose(transform_list)
