import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
import os
from PIL import Image

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, size=256):
        self.root_dir = root_dir
        self.size = size
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            # Handle corrupted images or non-image files by loading the next one
            return self.__getitem__((idx + 1) % len(self))
            
        img = np.array(img)
        
        # Resize to a standard size
        img = resize(img, (self.size, self.size), anti_aliasing=True)

        # Convert to Lab color space
        # L channel is for lightness (grayscale), a and b are for color
        lab_img = rgb2lab(img)

        # Separate channels
        # L is input, ab is the target we want to predict
        l_channel = lab_img[:, :, 0]
        ab_channels = lab_img[:, :, 1:]

        # Normalize L channel to be between -1 and 1 (from 0-100)
        l_channel = (l_channel / 50.0) - 1.0
        # Normalize ab channels to be between -1 and 1 (from -128 to 127)
        ab_channels = ab_channels / 128.0

        # Convert to tensors
        # PyTorch expects channels first: (C, H, W)
        l_tensor = torch.from_numpy(l_channel).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_channels).permute(2, 0, 1).float()

        return {'L': l_tensor, 'ab': ab_tensor}

def get_dataloader(root_dir, batch_size=16, shuffle=True, num_workers=4):
    dataset = ColorizationDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Utility function to convert model output back to a displayable image
def lab_to_rgb_image(l_channel, ab_channels):
    """
    Converts L and ab channels back to an RGB image.
    l_channel: (1, H, W) tensor, normalized -1 to 1
    ab_channels: (2, H, W) tensor, normalized -1 to 1
    """
    # Denormalize
    l = (l_channel.squeeze(0).cpu().numpy() + 1.0) * 50.0
    ab = ab_channels.permute(1, 2, 0).cpu().numpy() * 128.0

    # Combine channels
    lab_img = np.zeros((l.shape[0], l.shape[1], 3))
    lab_img[:, :, 0] = l
    lab_img[:, :, 1:] = ab

    # Convert to RGB
    rgb_img = lab2rgb(lab_img)
    return (rgb_img * 255).astype(np.uint8)
