import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import nibabel as nib
import torchio.transforms as transforms
from skimage.transform import resize
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    CropForegroundd,
    Resized
)
from monai.data import Dataset, DataLoader
from monai.utils import first

class CustomDataset(data.Dataset):
    """Dataset class for the dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the dataset."""
        self.image_dir = image_dir
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
        filename, label = dataset[index]        
        image = nib.load(filename).get_fdata()
        image = resize(image, (image.shape[0], image.shape[1], image.shape[2]), mode='constant')
        image = torch.from_numpy(image).float().view(1, image.shape[0], image.shape[1], image.shape[2])
        print(image.shape)
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_loader(image_dir, image_depth, image_size=256, batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(transforms.RandomFlip(axes=('LR',)))
    transform.append(transforms.CropOrPad((image_size, image_size, image_depth)))
    # transform.append(T.ToTensor())
    # transform.append(transforms.ZNormalization(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CustomDataset(image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader