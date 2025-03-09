import os
import tarfile
import torch.utils.data as data
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import download_url
import transforms.transforms as extended_transforms
from transforms.organize_transform import make_transform, magnitude_dyada


def custom_voc_cmap(N=256, normalized=False): 
        color_map = {
            0: [255, 255, 255],   # OTR     
            1: [160, 32, 240],    # DEB
            2: [0, 0, 255],       # LYM     
            3: [255, 255, 0],     # NORM
            4: [255, 97, 0],      # STR
            5: [255, 0, 0]        # TUM
        }

        cmap = np.zeros((N, 3), dtype='float32' if normalized else 'uint8')
        for i in range(N):
            if i in color_map:
                cmap[i] = np.array(color_map[i])
            else:
                cmap[i] = np.array([0, 0, 0])  
        if normalized:
            cmap = cmap / 255.0
        return cmap

class OurSegmentation(data.Dataset):
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 magnitude=1):

        self.root = os.path.expanduser(root)
        self.transform = transform  
        self.target_aux_transform = extended_transforms.MaskToTensor()
        self.image_set = image_set
        self.magnitude = magnitude
        self.MAGNITUDE = torch.zeros(50000) 
        self.magnitude_transform = magnitude_dyada
        self.is_warmup_finished = False
        self.warmup_transform = make_transform(magnitude=self.magnitude)[0] 
        
        
        self.target_transform = extended_transforms.MaskToTensor()
        voc_root = os.path.join(self.root, image_set)
        image_dir = os.path.join(voc_root, 'img')

        mask_dir = os.path.join(voc_root, 'mask')
        split_f = os.path.join(voc_root, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x) for x in file_names]
        self.masks = [os.path.join(mask_dir, x) for x in file_names]
        assert (len(self.images) == len(self.masks))

    def set_transform(self, magnitude):
        transform_list = []
        for i in range(len(magnitude)):
            transform_list.append(self.magnitude_transform(magnitude=self.magnitude)) 
        self.transform_list = transform_list
        return 
    def set_MAGNITUDE(self, index, magnitude):
        self.MAGNITUDE[index] = magnitude 
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        img_name = os.path.splitext(os.path.basename(self.images[index]))[0]
        target = Image.open(self.masks[index])

        if self.transform is not None: 
            img, target = self.transform(img, target) 
        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(target)
        if self.image_set == 'val':
            return img, target, img_name, mask_aux
        
        if self.is_warmup_finished:
            if self.MAGNITUDE[index] == 0:
                self.MAGNITUDE[index] = torch.tensor(0.1, device=self.MAGNITUDE.device)
            t = self.magnitude_transform(magnitude=self.MAGNITUDE[index].item())
            img_color = t(img)
        else:   
            img_color = self.warmup_transform(img)
        return img, img_color, target, img_name, mask_aux, index


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

class TestSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = custom_voc_cmap()
    def __init__(self,
                 root,
                 image_set='test',
                 transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_aux_transform = extended_transforms.MaskToTensor()
        self.image_set = image_set
        self.target_transform = extended_transforms.MaskToTensor()
        image_dir = os.path.join(self.root, 'img')

        mask_dir = os.path.join(self.root, 'mask')

        split_f = os.path.join(self.root, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x) for x in file_names]
        self.masks = [os.path.join(mask_dir, x) for x in file_names]
        assert (len(self.images) == len(self.masks))
    def set_MAGNITUDE(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        img_name = os.path.splitext(os.path.basename(self.images[index]))[0]
        target = self.target_transform(target)

        if self.transform is not None:
            img, target = self.transform(img, target)
        mask_aux = self.target_aux_transform(target).cuda()

        return img, target, img_name, mask_aux

    
    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)