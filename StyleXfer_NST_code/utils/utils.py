# Here in this file, we will define the custom dataset actually 
# Custom dataset is actually a class which inherits the dataset class

from torch.utils.data import Dataset
# Purpose: Load images from a folder and prepare them for training.
# Steps:
# __init__: Takes a root directory (root) and an optional transform. Lists all files in the folder and filters only image files (.jpg, .png, .jpeg).
# __len__: Returns the number of images in the dataset.
# __getitem__: Loads an image at a given index. Converts it to RGB. Applies transforms if provided. Returns the processed image.
# This makes your dataset compatible with PyTorch’s DataLoader.

import os
from PIL import Image, ImageFile
# Loads the Image class from the Pillow library (PIL = Python Imaging Library).
# Provides tools to open, manipulate, and convert images. In your dataset:
# Image.open(image_path) → opens an image file.
# .convert('RGB') → ensures the image is in RGB format (3 color channels), which is standard for training.

# Imports Python’s built‑in warnings module, which lets you control how warning messages are displayed (show, hide, or filter them).
import warnings

# By default, Pillow refuses to load images that are incomplete (truncated).
# Setting this to True tells Pillow: “Even if the file is missing some bytes, try to load it anyway.”
# This prevents OSError: image file is truncated from stopping your training loop.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Image.MAX_IMAGE_PIXELS = None
# Pillow has a built‑in safety limit (~178 million pixels) to block extremely large images, because they could be used in denial‑of‑service attacks.
# Setting this to None removes the limit, so Pillow will open images of any size.
# You then control memory safety yourself by resizing them (e.g., with thumbnail
Image.MAX_IMAGE_PIXELS = None

# Even if you disable the pixel limit, Pillow will still issue warnings when an image is very large.
# This line tells Python’s warnings system to ignore those specific warnings, so your logs stay clean.
# Training continues without cluttered output.
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


# Here we are importing the image transformation utilities from PyTorch’s torchvision library. 
from torchvision import transforms 
# What torchvision Is :-
# torchvision is a companion library to PyTorch, focused on computer vision tasks.
# It provides:
# Pretrained models (ResNet, VGG, etc.).
# Common datasets (CIFAR, ImageNet, COCO).
# Image utilities (loading, saving, transforming).

# What transforms Does :-
# transforms is a submodule inside torchvision that handles image preprocessing and augmentation.
# It lets you build pipelines that convert raw images into tensors suitable for training.

# Common Transforms :-
# Some of the most frequently used ones include:
# transforms.Resize((H, W)) → resizes image to given dimensions.
# transforms.RandomCrop(size) → randomly crops image (augmentation).
# transforms.CenterCrop(size) → crops from the center.
# transforms.ToTensor() → converts a PIL image or NumPy array into a PyTorch tensor (scales pixel values to [0,1]).
# transforms.Normalize(mean, std) → normalizes pixel values channel‑wise.
# transforms.ColorJitter() → randomly changes brightness, contrast, saturation, hue.



# This class is a custom PyTorch dataset loader designed to read images from a folder and make them usable in training pipelines.
# Inherits from torch.utils.data.Dataset, which is the base class for all datasets in PyTorch.
class ImageFolderDataset(Dataset):
    # __init__(self, root, transform=None) :- Runs when you create an instance of the dataset.
    # root: the folder path containing images.
    # transform: optional preprocessing (resize, crop, convert to tensor, normalize, etc.).
    # self.files = list(os.listdir(root)): lists all files in the folder.
    # self.files = [p for p in self.files if p.endswith(...)]: keeps only image files (.jpg, .png, .jpeg).
    # This sets up the dataset by collecting all valid image paths.
    def __init__(self, root, transform = None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', '.jpeg'))]


    # Returns the number of images in the dataset.
    # This is required by PyTorch’s DataLoader so it knows how many batches to create.
    def __len__(self):
        return len(self.files)


    # __getitem__(self, idx) :- Defines how to fetch one sample (image) by index.
    # Steps:
    # Build the full path: os.path.join(self.root, self.files[idx]).
    # Open the image with PIL: Image.open(...).convert('RGB').
    # Apply transforms if provided (resize, crop, tensor conversion, etc.).
    # Return the processed image.
    # This makes the dataset iterable — you can loop through it or feed it into a DataLoader.
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        
        try:
            # Try to open the image safely
            image = Image.open(image_path).convert('RGB')

            # Resize oversized images to a safe max size
            max_size = (1024, 1024)   # adjust as needed

            # image.thumbnail(max_size, Image.Resampling.LANCZOS) :- Resizes the image so that it fits within the 1024×1024 box.
            # Preserves the aspect ratio (no stretching). Uses the LANCZOS filter, which is a high‑quality resampling method that produces sharp, smooth results when downscaling.
            # Example: If the image is 4000×3000 → it becomes 1024×768. If the image is 800×600 → it stays 800×600 (since it already fits).
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        except Exception as e:
            # If Pillow fails to open or process the image (e.g., file is truncated, corrupted, or unreadable), the code jumps here.
            # If the image is corrupted or unreadable, skip it
            print(f"Skipping bad image: {image_path}, error: {e}")

            # Instead of crashing, it skips the bad image and tries the next one.
            # (idx + 1) % len(self.files) ensures it wraps around safely if you’re at the last image. This way, training continues without interruption.
            return self.__getitem__((idx + 1) % len(self.files))

        if self.transform:
            # So after applying transformations, we will get the tensor value of the image because one of the transforms if converting image to tensor actually
            image = self.transform(image)
            # Instead of a PIL image, you now have a tensor of shape: [𝐶,𝐻,𝑊]
            # where:
            #     - C = number of channels (3 for RGB).
            #     - H = height (final_size).
            #     - W = width (final_size).
            # Pixel values are scaled to the range [0, 1] (float32).

        return image




# This function is used to apply some transformations on the image
# size: an initial resize dimension (optional).
# crop: boolean flag (True/False) — whether to crop or not.
# final_size: the target resolution for the image.
# transform_list: a list that will hold all transformations.
def get_transform(size, crop, final_size):
    transform_list = []

    # If size is greater than 0, the image is resized to that dimension first.
    # Example: size=256 → image resized to 256 pixels on the shorter side.
    if size > 0:
        transform_list.append(transforms.Resize(size))
    
    # If crop=True → randomly crops the image to final_size.
    # If crop=False → directly resizes the image to final_size.
    # e.g crop=True, final_size=128 → randomly crop a 128×128 patch. crop=False, final_size=128 → resize entire image to 128×128.
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    # Converts the image (PIL or NumPy) into a PyTorch tensor. Pixel values are scaled to [0,1].
    # here we are converting the image to tensors because we will actually perform all the operations of NST on tensors only
    transform_list.append(transforms.ToTensor())

    # Chains all transformations together into one callable pipeline. So when you pass an image, it goes through resize → crop/resize → tensor conversion in sequence.
    return transforms.Compose(transform_list)





# This function is the core of Adaptive Instance Normalization (AdaIN), which is widely used in neural style transfer.
# The goal is to re-style the content features so that their statistical distribution (mean and variance) matches that of the style features. This allows the content image to adopt the "style" of another image while keeping its structure.
def adaptive_instance_normalization(content_feat, style_feat):
    # Both content_feat and style_feat are 4D tensors: [𝑁,𝐶,𝐻,𝑊] → batch size, channels, height, width.
    # size stores the shape for later broadcasting.
    # [batch size, channels, h, w]
    size = content_feat.size()

    # For each channel, compute mean and standard deviation across spatial dimensions.
    # calc_mean_std returns per-channel statistics:
    # style_mean, style_std → style distribution
    # content_mean, content_std → content distribution
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # Normalize the content features:
    # Subtract the content mean → centers distribution at 0
    # Divide by content std → scales distribution to unit variance
    # Result: content features now have zero mean and unit variance.
    # content_mean.expand(size) :- content_mean was computed earlier as [batch_size, channels, 1, 1].
    # .expand(size) broadcasts it to the same shape as content_feat → [batch_size, channels, height, width].
    # Now each pixel in a channel can be directly compared to that channel’s mean.
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    # Rescale normalized content features to match the style statistics:
    # Multiply by style_std → gives same variance as style
    # Add style_mean → shifts mean to match style
    # Final output: content features transformed to have style’s mean and variance.
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)





# This function calc_mean_std is a helper that computes the channel-wise mean and standard deviation of a feature map tensor. These statistics are essential for Adaptive Instance Normalization (AdaIN). 
def calc_mean_std(feat, eps=1e-5):
    # feat is a 4D tensor: [𝑁,𝐶,𝐻,𝑊]
    # batch_size = number of images in the batch.
    # channels = number of feature channels.
    # [batch size, channels, h, w]
    size = feat.size()


    # The assertion ensures the tensor has 4 dimensions.
    # size = feat.size() gives the shape of the tensor (e.g. [N, C, H, W]).
    # len(size) is the number of dimensions.
    # The assertion ensures the input tensor has exactly 4 dimensions:
    # Batch size (N), Channels (C), Height (H) & Width (W)
    # If the tensor isn’t 4D, the program will raise an error immediately. This is a safeguard because the function is designed only for 4D feature maps.
    assert (len(size) == 4)

    batch_size, channels = size[:2]

    # feat has shape [batch_size, channels, height, width].
    # feat.view(batch_size, channels, -1) reshapes it into [batch_size, channels, H*W].
    # Each channel’s spatial values (pixels/features) are now in a single vector.
    # e.g If feat = [2, 3, 4, 4] → after .view(2, 3, -1) → [2, 3, 16].
    # .mean(dim=2) → average across the flattened spatial dimension (H*W). As dim=0 → batch (N), dim=1 → channels (C) & dim=2 → flattened spatial dimension (H × W)
    # Result shape: [batch_size, channels] → one mean per channel per image.
    # .view(batch_size, channels, 1, 1) reshapes it back i.e of 4D shape, so it can be broadcasted later.
    # feat_mean = per-channel mean, ready for broadcasting.
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    # .var(dim=2, unbiased=False) → variance across spatial positions.
    # unbiased=False means dividing by 𝑛 instead of 𝑛−1 (slightly different statistical convention, but common in deep learning).
    # + eps (epsilon value) adds a tiny constant (1e-5) to avoid division by zero or numerical instability.
    # feat_var = per-channel variance.
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    
    # Square root of variance → standard deviation. Reshaped for broadcasting.
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    
    return feat_mean, feat_std


