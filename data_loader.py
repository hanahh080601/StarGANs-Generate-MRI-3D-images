import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import nibabel as nib
import math
import torchio as tio
import torchio.transforms as transforms
from skimage.transform import resize

class CustomDataset(data.Dataset):
    """Dataset class for the dataset."""

    def __init__(self, image_dir, patch_size, stride, image_size, transform, mode):
        """Initialize and preprocess the dataset."""
        self.image_dir = image_dir
        self.image_depth = 240
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the dataset."""
        all_attr_names = list(os.listdir(os.path.join(self.image_dir, self.mode)))
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        for attr_name in all_attr_names:
            for i, filename in enumerate(os.listdir(os.path.join(self.image_dir, self.mode, attr_name))):
                label = np.zeros((len(all_attr_names))).tolist()
                label[self.attr2idx[attr_name]] = 1
                image_path = os.path.join(self.image_dir, self.mode, attr_name, filename)
                if self.mode == 'train':
                    self.train_dataset.append([image_path, label])
                else:
                    self.test_dataset.append([image_path, label])

        print('Finished preprocessing the dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset

        num_patches = math.ceil((self.image_depth - self.patch_size + self.stride) / self.stride) ** 3

        filename, label = dataset[int(index//num_patches)]        
        image = nib.load(filename).get_fdata()
        image = resize(image, (self.image_size, self.image_size, self.image_depth), mode='constant', preserve_range=True)
        image = np.expand_dims(image, axis=0)  # add channel dimension

        patches = []
        for i in range(0, self.image_depth - self.patch_size + 1, self.stride):
            for j in range(0, self.image_size - self.patch_size + 1, self.stride):
                for k in range(0, self.image_size - self.patch_size + 1, self.stride):
                    patch = image[:, i:i+self.patch_size, j:j+self.patch_size, k:k+self.patch_size]
                    patch = self.transform(patch)
                    patches.append(patch)
        return patches[int(index%num_patches)], torch.FloatTensor(label)


    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, patch_size, stride, image_size, batch_size, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(tio.ToCanonical())
    transform.append(tio.RescaleIntensity(out_min_max=(0, 1)))
    if mode == 'train':
        transform.append(tio.RandomFlip(axes=(0,)))
    transform = tio.Compose(transform)

    dataset = CustomDataset(image_dir, patch_size, stride, image_size, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader