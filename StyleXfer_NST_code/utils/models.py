# In this file, we will define the core encoder–decoder architecture used in our AdaIN Neural Style Transfer project. 
# Here we will create two classes:-
# 1. VGGEncoder Class :-
# Purpose: Extracts hierarchical feature maps from an image using a pretrained VGG network.
# Uses a pretrained VGG network (trained on ImageNet) to extract hierarchical features from images.
# These features capture content structure and style patterns at different levels (edges, textures, shapes, semantics).


# 2. Decoder Class :-
# Purpose: Reconstructs an image back from encoded features after AdaIN has blended content + style.

# How They Work Together :-
# Encoder (VGGEncoder) :
# - Input: content image + style image.
# - Output: feature maps at different levels.

# AdaIN (Adaptive Instance Normalization) :
# - Aligns content features with style statistics (mean & std).

# Decoder (Decoder) :
# - Input: AdaIN‑modified features.
# - Output: stylized image in RGB space.


import torch

# It brings in PyTorch’s neural network module (torch.nn) and gives it the shorthand name nn.
import torch.nn as nn

# What torch.nn Provides :-
# torch.nn is the toolbox for building neural networks in PyTorch. It contains:
# Layers :
# nn.Linear → fully connected (dense) layer.
# nn.Conv2d → convolutional layer for images.
# nn.LSTM, nn.GRU → recurrent layers for sequences.

# Activations :
# nn.ReLU, nn.Sigmoid, nn.Softmax, etc.

# Loss Functions :
# nn.CrossEntropyLoss, nn.MSELoss, etc.

# Model Container :
# nn.Module → the base class for all models and layers.
# You subclass nn.Module to define custom models. 


# Here this encoder remains fixed i.e we will use pre-trained VGG model only & will not perform any training for it.
# But for decoder we will perform training i.e we will actually trained that.



# What encode does :-
# Uses a pretrained VGG network (trained on ImageNet) to extract hierarchical features from images.
# These features capture content structure and style patterns at different levels (edges, textures, shapes, semantics).

# Why it’s fixed :-
# VGG is already trained to recognize rich visual features.
# In style transfer, we don’t want to retrain it — we only use it as a feature extractor backbone.
# That’s why requires_grad=False is set: no gradient updates, no training.

# Output :-
# Multi‑level feature maps (h1, h2, h3, h4) or just the deepest features (h4 if is_test=True).

# This is the encoder class
# VGG (Visual Geometry Group network) is a famous CNN architecture used for image recognition and feature extraction
# It’s built from repeated blocks of:
# Convolutions (nn.Conv2d) → learn filters like edges, textures, shapes.
# ReLU activations (nn.ReLU) → add non‑linearity.
# Pooling (nn.MaxPool2d) → downsample feature maps.
# Padding (nn.ReflectionPad2d) → preserve spatial dimensions and reduce border artifacts.
class VGGEncoder(nn.Module):     #here we are inheriting this nn.Module class
    # vgg_path → an argument passed when you create the encoder, telling it where to load the pretrained VGG weights from.
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()     # here we are passing this ANN custom class as parameter for its parent class

        
        # Defining the architecture of our Convolutional Neural Network (CNN) using PyTorch’s nn.Sequential container.
        # nn.Sequential is a container that lets you stack layers in the order they should be applied.
        # Instead of writing a custom forward() method that manually calls each layer, you can just list them inside nn.Sequential, and PyTorch will apply them one after another.
        # THe disadvantage of using nn.Sequential is that as it is a container, we will only get the final result & not the intermediate results even of we want them
        # But fron VGG, we want those intermediate results actually
        self.vgg = nn.Sequential(
            # This here is actually the architecture of VGG
            # Here we are actually rebuilding the VGG network architecture layer by layer inside an nn.Sequential container
            # Here these relu1-1 or relu2-1 etc are defined by authors for taking the output from these layers when we want intermediate layers output actually
            # Block 1
            nn.Conv2d(3, 3, (1, 1)),
            # Reflection padding :- Used instead of zero padding to avoid sharp edges at borders, which is important for style transfer.
            # This reflection padding, the last values of edges gets copied as padding actually
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1 :- 1st relu of this block 1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2 :- 2nd relu of this block 1
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),    # Downsamples the feature maps by half (reduces spatial size, keeps important features).
            
            # Block 2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1 :-  1st relu of this block 2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2 :-  2nd relu of this block 2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            # Block 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1 :-  1st relu of this block 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2 :-  2nd relu of this block 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3 :-  3rd relu of this block 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4  :-  4th relu of this block 3
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            # Block 4
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            # so we will actually only take till this layer as we kind of want intermediate layer output which is these layers output 

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            # Block 5
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )

        # Now we will load the pre-trained weight for this VGG model from this file at vgg_path
        # so this file (i.e actually vgg_normalised.pth) actually contains the pre-trained weights of VGG 
        # Loads pretrained weights from the file at vgg_path to this VGG model
        self.vgg.load_state_dict(torch.load(vgg_path))
        # torch.load(vgg_path) :- Reads the file at vgg_path (e.g., "/vgg_normalized.pth").
        # That file contains a Python dictionary mapping layer names → parameter tensors (weights and biases).
        # self.vgg.load_state_dict(...) :- Loads those parameters into your self.vgg model (the nn.Sequential you defined).
        # Ensures each layer (Conv2d, ReLU, etc.) gets the correct pretrained values.
        # After this, your encoder is no longer randomly initialized — it’s using the pretrained VGG weights.

        # self.vgg.children() → returns all layers inside the nn.Sequential VGG model.
        # list(... )[:31] → keeps only the first 31 layers (up to relu4-1), since deeper layers are too abstract for style transfer.
        # nn.Sequential(*...) → rebuilds a new sequential model with just those layers.
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        # But currently this will give the output after all these layers from 0 till 31 & not give me the output of intermediate layers, so for that we will separate these layers into blocks
        
        # Deeper layers capture very high‑level semantics (e.g., “dog vs cat”), not fine visual details.
        # For style transfer, we need textures, colors, and mid‑level patterns, not classification features.
        # In AdaIN (and earlier NST papers), relu4-1 is chosen as the “sweet spot” for content features.
        # Shallower layers (conv1_x, conv2_x, conv3_x) are still used for style statistics, but the deepest block is avoided.
        # Think of VGG as a hierarchy of vision:
        # Early layers → detect edges, colors, textures.
        # Middle layers → detect shapes, motifs, patterns.
        # Deep layers → detect objects and categories
        # For style transfer, we care about textures + patterns + structure, not “object identity.” That’s why we stop at relu4-1.


        # Converts the truncated VGG layers into a Python list.
        # Now you can slice them into groups (blocks).
        enc_layers = list(self.vgg.children())

        # enc_1 (layers 0–3) → Block 1 (conv1_x): Low‑level features like edges and textures.
        # enc_2 (layers 4–10) → Block 2 (conv2_x): Mid‑level features like simple patterns.
        # enc_3 (layers 11–17) → Block 3 (conv3_x): Higher‑level features like shapes and object parts.
        # enc_4 (layers 18–30) → Block 4 (conv4_x): Semantic features (content structure). This is the last block used in AdaIN.
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        # Here these relu1-1 i.e 0 to 4 or relu2-1 etc are defined by authors for taking the output from these layers when we want intermediate layers output actually

        # Here we are actually freezing the encoder’s parameters so they don’t get updated during training
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            # getattr(self, name) → dynamically fetches each block like this (self.enc_1, self.enc_2, etc.).
            # .parameters() returns all trainable tensors (weights and biases) inside that block.
            for param in getattr(self, name).parameters():
                param.requires_grad = False    # tells PyTorch not to compute gradients for these parameters.
                # Without gradients, the optimizer won’t update them during training.


    # That forward method defines how data flows through our encoder when we call it.
    def forward(self, input, is_test=False):
        # The image tensor goes through the first block of VGG (conv1_x). Produces low‑level features (edges, textures). Stored in h1.
        h1 = self.enc_1(input)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        if is_test:   # i.e during testing time
            # Only returns h4 (deepest features).
            # Useful when you just want the final content representation (e.g., during evaluation).
            return h4
        
        # Returns all four feature maps (h1, h2, h3, h4).
        # Useful during training, since style transfer often compares statistics at multiple layers (shallow for style, deep for content).
        # SInce during training, we need to calculate loss & for that we need all these but during test, we do not need to calculate loss, so no need of all these & only need h4
        return h1, h2, h3, h4
    




# Decoder :- 
# It is actually the mirror image of the encoder
# What it does :-
# Takes the AdaIN‑modified features (content aligned with style statistics).
# Reconstructs them back into a full RGB image.
# Uses upsampling + convolution layers to progressively build the image.

# Why it’s trained :-
# Unlike VGG, the decoder starts untrained.
# It must learn how to invert VGG features back into pixel space while preserving the blended style.
# Training teaches it to generate visually coherent stylized images.

# Output :-
# A 3‑channel RGB image (the stylized result).


#  In Decoder, instead of using MaxPool2d layer, we actually uses UpSampling layer
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # Adds a 1‑pixel border around the feature map by reflecting edge values. Prevents sharp artifacts compared to zero padding. Important in style transfer to keep smooth edges.
            nn.ReflectionPad2d((1, 1, 1, 1)),

            # Convolution layer: reduces channels from 512 → 256. Learns filters to start reconstructing finer details. Kernel size 3×3 captures local patterns.
            nn.Conv2d(512, 256, (3, 3)),

            # Non‑linear activation. Ensures the network can learn complex mappings, not just linear ones.
            nn.ReLU(),

            # Doubles the spatial resolution (height × width). Gradually reconstructs the image size back to original. “Nearest” means pixels are duplicated (simple but effective).
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),         
        )

    # Now here no need to assign weights & all because currently this is untrained decoder & we need to actually train this model


    # input here is the encoded feature map (after AdaIN has blended content + style).
    def forward(self, input):
        # Return this output which is a 3‑channel RGB image tensor. This is the reconstructed stylized image, generated from the 
        return self.net(input)
        # self.net(input) :- self.net is the decoder architecture you defined earlier using nn.Sequential.
        # It contains all the layers: reflection padding, convolutions, ReLU activations, and upsampling.
        # When you call self.net(input), PyTorch automatically passes the input through each layer in order.
