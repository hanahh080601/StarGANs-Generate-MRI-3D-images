import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import nibabel as nib
import torchio as tio
import torchio.transforms as transforms
from skimage.transform import resize

class CustomDataset(data.Dataset):
    """Dataset class for the dataset."""

    def __init__(self, image_dir, image_depth, image_size, patch_size, stride,  transform, mode):
        """Initialize and preprocess the dataset."""
        self.image_dir = image_dir
        self.image_depth = image_depth
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
                break
            break

        print('Finished preprocessing the dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]        
        image = nib.load(filename).get_fdata()
        print("Origin image shape:", image.shape)
        image = resize(image, (self.image_size, self.image_size, self.image_depth), mode='constant')
        image = torch.from_numpy(image).float().view(1, self.image_depth, self.image_size, self.image_size)
        print(image.shape)
        # image = image.view(1, image.shape[2], image.shape[1], image.shape[0])

        patches_unpacked = image.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        patches = patches_unpacked.permute(1, 2, 3, 0, 4, 5, 6).reshape(-1, self.patch_size, self.patch_size, self.patch_size)
        transformed_patches = []
        # print(image.shape)
        for i in range(patches.shape[0]):
            patch = patches[i, :, :, :]
            patch = resize(patch, (patch.shape[0], patch.shape[1], patch.shape[2]), mode='constant')
            patch = torch.from_numpy(patch).float().view(1, patch.shape[0], patch.shape[1], patch.shape[2])
            transformed_patch = self.transform(patch)
            print(transformed_patch.shape)
            transformed_patches.append(transformed_patch)
        print(len(transformed_patches))
        return transformed_patches, torch.FloatTensor(label) * len(transformed_patches)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_loader(image_dir, patch_size, stride, image_depth=155, image_size=256, batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(transforms.RandomFlip(axes=('LR',)))
    transform.append(transforms.CropOrPad((patch_size, patch_size, patch_size)))
    # transform.append(transforms.Resize((image_size, image_size, image_depth)))
    # transform.append(T.ToTensor())
    transform.append(transforms.ZNormalization(masking_method=tio.ZNormalization.mean))
    transform = T.Compose(transform)

    dataset = CustomDataset(image_dir, patch_size, stride, image_depth, image_size, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader