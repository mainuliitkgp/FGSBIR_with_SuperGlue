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
import numpy as np
import cv2

from superglue_models.matching import Matching
from superglue_models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def read_image(image, device, resize, rotation, resize_float):
    #image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name , hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name)
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp) ## dict

        self.Train_Sketch = [x for x in self.Coordinate if 'train' in x] ## list of keys
        self.Test_Sketch = [x for x in self.Coordinate if 'test' in x]

        # Load the SuperPoint and SuperGlue models.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        self.config = {
            'superpoint': {
                'nms_radius': hp.nms_radius,
                'keypoint_threshold': hp.keypoint_threshold,
                'max_keypoints': hp.max_keypoints
            },
            'superglue': {
                'weights': hp.superglue,
                'sinkhorn_iterations': hp.sinkhorn_iterations,
                'match_threshold': hp.match_threshold,
            }
        }
        self.matching = Matching(self.config).eval().to(self.device)

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
            sketch_img = Image.fromarray(sketch_img).convert('L')

            positive_img_grey = Image.open(positive_path).convert('L')
            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img_grey = F.hflip(positive_img_grey)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            # Load the image pair.
            image0, inp0, scales0 = read_image(
                np.asarray(sketch_img), self.device, self.hp.resize, 0, self.hp.resize_float)
            image1, inp1, scales1 = read_image(
                np.asarray(positive_img_grey), self.device, self.hp.resize, 0, self.hp.resize_float)

            # Perform the matching.
            pred = self.matching({'image0': inp0, 'image1': inp1})
            sketch_img = pred['mlp_vec0'].view(1, -1)

            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'negative_img': negative_img, 'negative_path': negative_sample
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('L')

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img_grey = Image.open(positive_path).convert('L')
            positive_img = self.test_transform(Image.open(positive_path).convert('RGB'))

            # Load the image pair.
            image0, inp0, scales0 = read_image(
                np.asarray(sketch_img), self.device, self.hp.resize, 0, self.hp.resize_float)
            image1, inp1, scales1 = read_image(
                np.asarray(positive_img_grey), self.device, self.hp.resize, 0, self.hp.resize_float)

            # Perform the matching.
            pred = self.matching({'image0': inp0, 'image1': inp1})
            sketch_img = pred['mlp_vec0'].view(1, -1)

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
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True)

    dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False)

    return dataloader_Train, dataloader_Test

def get_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(299)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)
