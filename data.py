import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RailsDataset(Dataset):
	def __init__(self, items, image_dir, mask_dir, rail_code, transform=None):
		self.items = items
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.rails_code = rail_code
		self.transform = transform
	def __len__(self):
		return len(self.items)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.items[index])
		image = np.array(Image.open(img_path).convert("RGB"))
		mask_path = os.path.join(self.mask_dir, self.items[index].replace("jpg","png"))
		mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
		mask[mask != self.rails_code] = 0.0
		mask[mask == self.rails_code] = 1.0

		if self.transform is not None:
			augmentations = self.transform(image=image, mask=mask)
			image = augmentations["image"]
			mask = augmentations["mask"]
		return image, mask
class TestDataset(Dataset):
	def __init__(self, items, image_dir, transform=None):
		self.items = items
		self.image_dir = image_dir
		self.transform = transform

	def __len__(self):
		return len(self.items)

	def __getitem__(self, index):
		name = self.items[index]
		img_path = os.path.join(self.image_dir, name)
		image = np.array(Image.open(img_path).convert("RGB"))

		if self.transform is not None:
			augmentations = self.transform(image=image)
			image = augmentations["image"]
		return image, name
class MeasureDataset(Dataset):
	def __init__(self, items, image_dir, transform=None):
		self.items = items
		self.image_dir = image_dir
		self.transform = transform
	def __len__(self):
		return len(self.items)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.items[index])
		image = np.array(Image.open(img_path).convert("RGB"))

		if self.transform is not None:
			augmentations = self.transform(image)
		return augmentations
def get_loaders(
       train_items,
       val_items,
       img_dir,
       mask_dir,
       val_img_dir,
       val_mask_dir,
       batch_size,
       train_transform,
       val_transform,
       rail_code,
       num_workers=0,
       pin_memory=False
    ):
    train_ds = RailsDataset(
        items = train_items,
        image_dir = img_dir,
        mask_dir = mask_dir,
        rail_code=rail_code,
        transform = train_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )
    val_ds = RailsDataset(
        items = val_items,
        image_dir = val_img_dir,
        mask_dir = val_mask_dir,
        rail_code=rail_code,
        transform = val_transform
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    )
    return train_loader, val_loader
def get_test_loaders(
       items,
       img_dir,
       transform,
       num_workers=0,
       pin_memory=False
    ):
    ds = TestDataset(
        items = items,
        image_dir = img_dir,
        transform = transform
    )
    loader = DataLoader(
        ds,
        batch_size = 1,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    )
    return loader
def get_measure_loaders(
       items,
       img_dir,
       transform,
       num_workers=0,
       pin_memory=False
    ):
    ds = MeasureDataset(
        items = items,
        image_dir = img_dir,
        transform = transform
    )
    loader = DataLoader(
        ds,
        batch_size = 1,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    )
    return loader
