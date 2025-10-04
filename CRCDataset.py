import os.path
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms

IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
SHAPE = [1, 224, 224]

def pick_random_image_by_ext(folder_path: str,
                             exts=None,
                             seed=None):
    if exts is None:
        exts = IMAGE_EXTS
    if seed is not None:
        random.seed(seed)

    if not os.path.isdir(folder_path):
        return None

    candidates = []
    for fn in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fn)
        if os.path.isfile(full_path) and os.path.splitext(fn)[1].lower() in exts:
            candidates.append(full_path)

    if not candidates:
        return None
    else:
        return random.choice(candidates)

class MRIDataset(Dataset):
    def __init__(self,
                 image_path: str,
                 annotation_path: str,
                 transforms = transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor()
                 ])):
        self.samples = []
        self.trans = transforms
        csv_data = pd.read_csv(annotation_path)
        csv_data = csv_data.set_index('fname')
        for fname, _ in csv_data.iterrows():
            label = csv_data.loc[fname, 'label']
            patient = csv_data.loc[fname, 'patient']
            lesion_dir = os.path.join(image_path, 'png', f'class{label+1}', f'{patient:03d}', 'lesion')
            full_path = os.path.join(image_path, 'png_sum', f'{fname:04d}.png')
            self.samples.append((full_path,
                                 pick_random_image_by_ext(lesion_dir),
                                 label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, lesion_path, label = self.samples[index]
        image = Image.open(image_path)
        lesion_image = Image.open(lesion_path)
        if self.trans is not None:
            image = self.trans(image)
            lesion_image = self.trans(lesion_image)

        return image, lesion_image, label

    def get_proportions(self, num_classes):
        proportions = [0] * num_classes
        for _, _, label in self.samples:
            label = int(label)
            proportions[label] += 1
        return proportions

class RawDataset(Dataset):
    def __init__(self,
                 image_path: str,
                 annotation_path: str,
                 transforms = transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor()
                 ])):
        self.samples = []
        self.transforms = transforms
        csv_data = pd.read_csv(annotation_path)
        csv_data = csv_data.set_index('fname')
        for fname, _ in csv_data.iterrows():
            label = csv_data.loc[fname, 'label']
            full_path = os.path.join(image_path, 'png_sum', f'{fname:04d}.png')
            self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def get_proportions(self, num_classes):
        proportions = [0] * num_classes
        for _, label in self.samples:
            label = int(label)
            proportions[label] += 1
        return proportions

class RoIDataset(Dataset):
    def __init__(self,
                 image_path: str,
                 annotation_path: str,
                 transforms = transforms.Compose([
                     transforms.Resize([224, 224]),
                     transforms.ToTensor()
                 ])):
        self.samples = []
        self.transforms = transforms
        csv_data = pd.read_csv(annotation_path)
        csv_data = csv_data.set_index('fname')
        for fname, _ in csv_data.iterrows():
            patient = csv_data.loc[fname, 'patient']
            label = csv_data.loc[fname, 'label']
            lesion_dir = os.path.join(image_path, 'png', f'class{label+1}', f'{patient:03d}', 'lesion')
            for img_name in os.listdir(lesion_dir):
                full_path = os.path.join(lesion_dir, img_name)
                self.samples.append((full_path, label))
        self.samples = list(dict.fromkeys(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        label = int(label)
        image = Image.open(img_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def get_proportions(self, num_classes):
        proportions = [0] * num_classes
        for _, label in self.samples:
            label = int(label)
            proportions[label] += 1
        return proportions

class SubDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 indices,
                 transforms):
        self.base = base_dataset
        self.indices = indices
        self.transforms = transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if len(self.base[0]) == 2:
            img, label = self.base[self.indices[idx]]
            if self.transforms:
                img = self.transforms(img)

        elif len(self.base[0]) == 3:
            img, roi, label = self.base[self.indices[idx]]
            if self.transforms:
                img = self.transforms(img)
                roi = self.transforms(roi)

            

        return img, roi, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_dataset = MRIDataset(r'data/MRI/Images',
                               r'data/MRI/fname2label.csv')
    print(train_dataset.__len__())
    # print(train_dataset.get_proportions(num_classes=3))

