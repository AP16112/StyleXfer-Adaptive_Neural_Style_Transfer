# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Project :- StyleXfer – "Adaptive Neural Style Transfer"

# StyleXfer is an AI‑powered project built on Adaptive Instance Normalization (AdaIN) that seamlessly transfers artistic styles onto images. 
# By intelligently aligning feature statistics between content and style, it produces visually striking transformations while preserving the essence of the original image. 
# Designed to be lightweight, efficient, and creative, StyleXfer showcases the fusion of deep learning and digital artistry.
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Steps to run this file to train model on kaggle :-

# Here we will use Kaggle for training this model because Kaggle provide free 16GB GPU, which helps in faster training
# So for that we firstly need to create the zip file of this 'StyleXfer_NST_code' folder as all of our model related code is inside this folder only
# And then we will go to :- https://www.kaggle.com/code 
# Then we will create a new notebook & then we will upload our ZIP folder as the new Dataset for that notebook because in kaggle, even folder can be uploaded as dataset only
# And then we will copy the path of our main python file i.e train.py & then write this code in notebook cell :-
# !python </kaggle/input/datasets/arpitpal07/stylexfer-model-training/StyleXfer_NST_code/train.py>  --batch_size 4 --epochs 160 --experiment='final_experiment' --save_interval 15  --content_dir='/kaggle/input/datasets/arpitpal07/stylexfer-model-training/StyleXfer_NST_code/content_dataset'  --style_dir='/kaggle/input/datasets/arpitpal07/stylexfer-model-training/StyleXfer_NST_code/style_dataset'  --vgg='/kaggle/input/datasets/arpitpal07/stylexfer-model-training/StyleXfer_NST_code/vgg_normalised.pth'                              
# ANd we running this, we also need to go to sessions options -> accelerator -> select GPU P100 -> it will turn on the free GPU
# Then start the session by clicking on the start session button
# And run that cell ->  It will start the training of model

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Steps for proper training :-
# we will firstly run this file with these arguments:-
# python train.py --batch_size 16 --epochs 160 --experiment='final_training'
# And then we will change the batch_size to 8 & final_size to 512 for better images & style_weight to 10 & resume = True, so that we can start the training where we left lastly i.e from 160th epoch 
# So then we will run this file using these parameters :-
# python train.py --batch_size 8 --epochs 200 --experiment='final_training'  --final_size 512  --style_weight 10  --resume --decoder_path='experiment/final_training/decoder_160.pth' --optimizer_path='experiment/final_training/optimizer_160.pth' 


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this file, we will actually write the logic for training our model for NST.
# This file is essentially our training script for the AdaIN Neural Style Transfer project. 
# This file is our training driver script. It :-
# Loads datasets and models.
# Applies AdaIN to fuse content + style features.
# Trains the decoder to reconstruct stylized images.
# Logs progress and saves checkpoints + sample outputs.


import argparse    # importing Python’s argparse module, which is part of the standard library.
# It lets us define the command‑line arguments for our script (like --epochs, --batch_size, --content_dir).
# When we run our script from the terminal, we can pass in values without editing the code.
# e.g   python train.py --epochs 10 --batch_size 8 --content_dir ./data/content --style_dir ./data/style

import torch

# Here we are importing the DataLoader class from PyTorch’s torch.utils.data module
from torch.utils.data import DataLoader
# DataLoader is PyTorch’s utility for loading datasets efficiently. It wraps around a Dataset (like your ImageFolderDataset) and provides:
# Batching → splits data into mini‑batches (e.g., 16 images at a time).
# Shuffling → randomizes the order of samples each epoch.
# Parallel loading → can use multiple worker threads to load data faster.
# Iteration → makes datasets iterable in training loops.

# torch.optim is PyTorch’s package that contains different optimization algorithms (like SGD, Adam, RMSprop).
# These optimizers are used to update the trainable parameters of your model (in this case, the decoder) during training.
import torch.optim as optim   # here we are importing optimizer actually

from pathlib import Path


# Here we are importing everything i.e classes or functions etc from the utils.py file present inside the utils folder 
from StyleXfer_NST_code.utils.utils import *
# Here we are importing everything i.e classes or functions etc from the models.py file present inside the utils folder 
from StyleXfer_NST_code.utils.models import *

# tqdm is a Python library that gives you progress bars for loops
from tqdm import tqdm

# save_image :- A function that saves a PyTorch tensor as an image file (e.g., .png, .jpg).
# The tensor should have shape [C, H, W] (channels, height, width) or [B, C, H, W] (batch of images).
# It automatically converts the tensor values into pixel values and writes them to disk.
from torchvision.utils import save_image




# function that sets up command‑line argument handling for our script. 
# When called, it will return an object (args) containing the values the user passed in when running the script.
def parse_arguments():
    # Creates an ArgumentParser object from Python’s argparse module.
    parser = argparse.ArgumentParser()
    # It will hold all the arguments we define with parser.add_argument(...).

    # --content_dir :- This is the name of the argument you can pass when running your script.
    # e.g python train.py --content_dir /user/images/content
    # If you don’t provide it, the script will use this default value.
    # type=str :- Specifies that the argument must be a string (in this case, a file path). If you try to pass something invalid (like a number when a string is expected), argparse will throw an error.
    # help='Location of content dataset' :- This is the description shown when you run: python train.py --help
    # It tells the user what the argument is for.
    parser.add_argument('--content_dir', type=str, default='./content_data_examples', help='Location of content dataset')
    
    parser.add_argument('--style_dir', type=str, default='./style_data_examples', help='Location of style dataset')
    
    # Here we will use this pre-trained VGG model for extracting the feature maps from images.
    # Here we are defining a command‑line argument for your script that tells it where to find the pretrained VGG model file
    parser.add_argument('--vgg', type=str, default='vgg_normalised.pth', help='Location of pre-trained VGG')
    # A .pth file in PyTorch is simply a checkpoint file that stores model parameters (weights, biases, optimizer states, etc.) in a serialized format.
    # Inside .pth, it contains tensors saved with torch.save().
    # Depending on how you save it, a .pth file can hold:
    # Model weights (state_dict of a neural network).
    # Optimizer state (momentum, learning rate, etc.).
    # Entire checkpoint (model + optimizer + epoch info).

    # So, If we don’t explicitly pass --experiment, the script assumes the experiment name is "experiment1".
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')
    # The experiment name is used to create a save directory: save_dir = Path('experiment') / args.experiment

    # Here we are defining a command‑line argument that controls the output image size.
    # If you don’t explicitly pass --final_size, the script assumes the final image size is 512 pixels (usually width × height).
    parser.add_argument('--final_size', type=int, default=256, help='Size of final image')
    
    parser.add_argument('--content_size', type=int, default=512, help='Size of content image')
    
    parser.add_argument('--style_size', type=int, default=512, help='Size of style image')
    
    # Here we are defining a command‑line flag that controls whether images should be cropped during preprocessing
    # Since it uses action='store_true', we don’t need to provide a value — just including --crop sets it to True.
    # default=True :- Sets the default value to True if you don’t provide the flag. So even if you don’t type --crop, cropping will be enabled by default.
    parser.add_argument('--crop', action='store_true', default=True, help='Crop image')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

    # Here this default value is a common choice for training the decoder in AdaIN style transfer.
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay')

    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')

    parser.add_argument('--content_weight', type=float, default=1.0, help='Content weight')
    
    parser.add_argument('--style_weight', type=float, default=5, help='Style weight')

    # It is a command‑line argument for logging frequency in our training script
    # If you don’t specify --log_interval when running the script, it defaults to 1. Meaning: log after every batch by default.
    # log_interval determines how often training information is printed/logged (like losses).
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    
    # Here we are adding a command‑line argument for how often to save model checkpoints during training
    # If you don’t specify --save_interval, the script defaults to saving every 2 epochs. Meaning: after every 2 epochs, the model checkpoint will be saved automatically.
    # save_interval determines how frequently the model is saved during training.
    parser.add_argument('--save_interval', type=int, default=2, help='Save interval')

    parser.add_argument('--resume', action='store_true', default=False, help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None, help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None, help='Path to optimizer checkpoint')


    # It tells the argparse parser to read the actual command‑line arguments we passed when running the script i.e this train.py file.
    # e.g python train.py --epochs 10 --batch_size 8   →   parse_args() will grab those values and store them in an object.
    return parser.parse_args()






# Here in this main() fn, we will write all of our training code actually
def main():
    args = parse_arguments()
    # print(args)

    # torch.device(...) :- Creates a device object that tells PyTorch where tensors and models should live.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Later in our code, we can move models and data to this device:
    # model.to(device)
    # tensor.to(device)   

    # Here we are creating the folder path where our training outputs (checkpoints, logs, images) will be saved.
    save_dir = Path('experiment') / args.experiment
    # Path('experiment') :- It represents a filesystem path object (instead of just a plain string).
    # Here, it points to a base folder called "experiment".
    # / args.experiment :- The / operator is overloaded in pathlib to mean path joining.
    # So, os.path.join("experiment", args.experiment) ,you can simply write: Path("experiment") / args.experiment

    # exist_ok=True :- Means: “Don’t throw an error if the folder already exists.”
    save_dir.mkdir(exist_ok=True, parents=True)   # is creating the directory on disk
    # parents=True :- means “Create all parent directories if they don’t exist.”
    # Example: If you ask for experiment/experiment1, but experiment/ doesn’t exist yet, it will create both experiment/ and experiment/experiment1.
    # Without this, it would fail unless the parent folder already existed.


    # Saving arguments values
    # Saving all the command‑line arguments we passed into a text file, so we have a record of how the experiment was run.
    # The with statement ensures the file is properly closed after writing, even if errors occur.
    with open(save_dir/'args.txt', 'w') as args_file:
        # args is the Namespace object returned by parser.parse_args().
        # vars(args) converts it into a dictionary & .items() gives you key–value pairs (like ('epochs', 10)).
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')


    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    # Now to read the data using the custom dataset which we created inside utils.py file, we will firstly create the object of ImageFOlderDataset class
    # And then we will pass them to DataLoader, so that it can creates the batches from that
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)
    
    # Now we will create the dataloaders from these datasets, which will actually create the batches of images & then return those batches from these datasets
    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle = True, pin_memory=True, drop_last=True)
    # batch_size=args.batch_size :- Groups samples into batches of size args.batch_size. Example: if args.batch_size=16, each iteration returns a tensor of shape [16, 3, H, W].
    # shuffle=True :- Randomizes the order of samples each epoch. Prevents the model from memorizing the dataset order and improves generalization.
    # SO it means that after every epoch, our dataset gets shuffled which prevents the model from memorizing the dataset order after every epoch
    # pin_memory=True :- Allocates batches in page‑locked (pinned) memory. Speeds up data transfer from CPU → GPU. Useful when training on CUDA devices.
    # drop_last=True :- Drops the last batch if it’s smaller than batch_size.
    # Ensures all batches are the same size (important for some models that expect fixed batch dimensions).
    # Example: if you have 103 images and batch size 16 → Normally: last batch would have 7 images. With drop_last=True: last batch is discarded, so you only get 96 images (6 full batches).

    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle = True, pin_memory=True, drop_last=True)

    # We do not need shuffling after every epoch in case of test dataloaders.
    # Training vs Testing DataLoader :-
    ## Training DataLoader (shuffle=True) :-
    # - Shuffling is important during training because it prevents the model from memorizing the order of samples.
    # - It improves generalization by ensuring batches are different each epoch.
    ## Testing/Validation DataLoader (shuffle=False) :-
    # Here, we want deterministic evaluation.
    # The model should see the test samples in a fixed order every time.
    # This ensures reproducibility — the same inputs always produce the same outputs.

    # Why No Shuffling in Testing :-
    # If you shuffle test data, results may vary slightly between runs (different batch composition).
    # That makes it harder to compare metrics (accuracy, loss, etc.) across experiments.    
    # Keeping the order fixed ensures consistent evaluation.

    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))

    # for batch in style_dataloader:
        # print(batch.shape)    # here we are usinf .shape because batch are also dataset only but of smaller size as comapred to original dataset

    encoder = VGGEncoder(args.vgg).to(device)

    # The decoder is the trainable network that reconstructs stylized images from encoded features.
    # .to(device) moves the model to the specified hardware device. This ensures all computations (forward pass, training) happen on the chosen device
    decoder = Decoder().to(device)

    # Now we will use optimizer to update the weights 
    # optim.Adam :- Chooses the Adam optimizer, one of the most popular algorithms in deep learning.
    # Adam combines the benefits of momentum (like SGD with momentum) and adaptive learning rates (like RMSprop). It’s well‑suited for training deep networks such as your decoder.
    # decoder.parameters() :- Passes all the trainable parameters (weights and biases) of the decoder into the optimizer.
    # Since the encoder is frozen (requires_grad=False), only the decoder’s parameters will be updated.
    # lr=args.lr :- Sets the learning rate (step size for updates).
    # args.lr means the learning rate is taken from command‑line arguments or a config file (so you can easily change it without editing code).
    # Typical values for AdaIN style transfer are around 1e-4.
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    
    # Learning rate decay is a technique to reduce the learning rate over time.
    # Early in training → larger steps (fast learning).
    # Later in training → smaller steps (fine‑tuning). 
    # Prevents overshooting and helps stabilize convergence.
    # In AdaIN style transfer, this ensures the decoder learns steadily without destabilizing the pretrained encoder features.
    
    # Here we are setting up a learning rate scheduler in PyTorch — a mechanism to automatically adjust the learning rate as training progresses. 
    # optim.lr_scheduler.LambdaLR :- A PyTorch scheduler that lets you define a custom function (lr_lambda) to control how the learning rate changes over epochs.
    # It wraps around your optimizer and modifies its learning rate at each step.
    # lr_lambda = lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch) :- Defines the decay function for the learning rate.
    # At epoch 0 → factor = 1.0 / (1.0 + 0) = 1.0 (no decay).
    # At epoch 1 → factor = 1.0 / (1.0 + args.lr_decay * 1).
    # At epoch N → factor = 1.0 / (1.0 + args.lr_decay * N).
    # This means the learning rate shrinks gradually as epochs increase.
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch:  1.0 / (1.0 + args.lr_decay * epoch)
    )


    # here we are handling the resume training functionality — it reloads saved model and optimizer states so you can continue training from where you left off.
    if args.resume:   # Checks if the --resume flag was passed when running the script.
        # If true, it will reload previously saved training progress.
        # decoder.load_state_dict(torch.load(args.decoder_path)) :- Loads the decoder’s weights from a checkpoint file (args.decoder_path).
        # torch.load(...) reads the saved state dictionary (layer weights and biases).
        # load_state_dict(...) restores those weights into the decoder model.
        # This means the decoder doesn’t start from scratch — it continues from where it was last saved.
        decoder.load_state_dict(torch.load(args.decoder_path))
        
        # Loads the optimizer’s state (Adam’s internal parameters like momentum buffers, learning rate schedule, etc.).
        # Ensures the optimizer resumes with the same learning dynamics as before.
        # Without this, even if the decoder weights are restored, the optimizer would “forget” its progress and restart fresh.
        optimizer.load_state_dict(torch.load(args.optimizer_path))

        # Why It’s Important :-
        # Training style transfer models can take hours or days.
        # You don’t want to lose progress if training is interrupted.
        # By saving both:
        # Decoder weights → the model’s learned knowledge.
        # Optimizer state → the training momentum and learning rate adjustments.
        # You can resume training seamlessly, as if nothing was interrupted.


    print('Training...')

    # Now for training loop, we need to firstly calculate the loss
    # And for that we will use mean squared loss actually
    # In style transfer:
    # Often used to compare feature maps (content loss) or Gram matrices (style loss).
    # It penalizes large differences more strongly, encouraging the decoder to reconstruct images close to the target.
    mse_loss = torch.nn.MSELoss()

    # Sets the encoder (VGG) to evaluation mode.
    # In PyTorch, models can be in:
    # Training mode (model.train()) → layers like dropout and batch normalization behave differently (they update statistics).
    # Evaluation mode (model.eval()) → those layers stop updating and use fixed behavior.
    # Since the encoder is pretrained and frozen, we don’t want it to change during training.
    # eval() ensures it acts purely as a fixed feature extractor.
    encoder.eval()

    running_loss = None   # total loss
    running_closs = None   # content loss
    running_sloss = None   # style loss

    # Now we will setup the training loop with a progress bar
    for epoch in range(args.epochs):
        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=min(len(content_dataloader), len(style_dataloader))
        )
        # zip(content_dataloader, style_dataloader) :- Combines the two dataloaders (content images and style images) into pairs.
        # Each iteration gives you one batch of content images and one batch of style images.
        # e.g for content_batch, style_batch in zip(content_dataloader, style_dataloader):
        # tqdm(...) :- Wraps the loop with a progress bar.
        # Shows how many batches have been processed, speed, and estimated time remaining.
        # total=min(len(content_dataloader), len(style_dataloader)) :- Ensures the progress bar length matches the smaller of the two datasets.
        # Prevents errors if content and style datasets have different sizes.
        # Training stops when the shorter dataloader runs out of batches


        running_loss = 0
        running_closs = 0
        running_sloss = 0

        # Iterates over the progress bar created earlier (tqdm(zip(content_dataloader, style_dataloader))).
        # Each iteration gives you:
        # content_batch: a batch of content images.
        # style_batch: a batch of style images.
        # These are paired together so the model can apply style transfer.
        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # Now firstly every image needs to pass through the encoder to get its feature map
            # encoder(content_batch) :- Passes the batch of content images through the pretrained VGG encoder. The encoder outputs feature maps (multi‑level representations of the image).
            # These features capture the structure, shapes, and semantic content of the image. Stored in c_feats.
            c_feats = encoder(content_batch)
            # encoder(style_batch) :- Passes the batch of style images through the same encoder.
            # Outputs feature maps that capture textures, colors, and patterns of the style image. Stored in s_feats.
            s_feats = encoder(style_batch)

            # So here these 'c_feats' & 's_feats' will be actually tuple of feature maps
            # print(len(c_feats))
            # print(len(s_feats))
            # print(type(c_feats))
            # print(c_feats)
            # print(c_feats[0].shape)


            # Here this 'adaptive_instance_normalization' will actually apply the AdaIN layer on these feature maps i.e we are passing the outputs of encoder to this AdaIN layer as per NST using AdaIN Architecture algorithm
            # c_feats and s_feats are lists (or tuples) of feature maps from the encoder at different layers.
            # [-1] selects the deepest feature map (usually from relu4_1 in VGG).
            # These are the most semantically rich features:
            # c_feats[-1] → content structure. So c_feats[-1] selects the deepest feature map (the one from the last layer the encoder outputs).
            # s_feats[-1] → style texture/color statistics.
            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])
            # adaptive_instance_normalization(c_feats[-1], s_feats[-1]) :- AdaIN aligns the mean and variance of the content features with those of the style features, so that we can say that style gets transfer to content image
            # AdaIN(𝑐,𝑠) = (𝜎(𝑠) ⋅ ((𝑐 − 𝜇(𝑐))/ 𝜎(𝑐))) + 𝜇(𝑠)
            # Intuition:
            # Normalize content features → remove their original style.
            # Re‑scale and re‑center them using style statistics → inject style appearance.
            # Result: t is a tensor of blended features (content structure + style appearance).


            # The decoder takes the blended features t and reconstructs them back into an RGB image.
            # This output g is the stylized image:
            # Preserves the layout/structure of the content image.
            # Painted with the textures/colors of the style image.
            g = decoder(t)    # it represent the output i.e generated image

            # Now as decoder is trainable, so we need to update its weights & for that we need to firstly find the loss
            # And for loss, we need feature map of this g, so we will pass it through encoder
            g_feats = encoder(g)

            # g_feats[-1] :- These are the features of the generated image (g) extracted by the encoder.
            # [-1] means the deepest feature map (high‑level representation of the generated image).
            # t  :- This is the target blended feature map produced by AdaIN. It represents the content structure aligned with the style statistics.
            loss_c = mse_loss(g_feats[-1], t) * args.content_weight
            # mse_loss(g_feats[-1], t) :- Computes the Mean Squared Error (MSE) between:
            # The generated image’s features (g_feats[-1]). And the The target AdaIN features (t).
            # This measures how close the generated image is to the desired blended representation.
            # Here t is actually shows the feature maps og content imahe which is slightly gets transformed
            # Although, instead of t, we can use c_feats, but authors found that using t gives better results, that's why we are using it here.
            # * args.content_weight :- Scales the loss by a user‑defined weight (--content_weight).
            # Allows you to control the balance between content preservation and style transfer:
            # Higher content weight → generated image sticks more closely to the original content structure.
            # Lower content weight → style dominates more strongly.

            # Now we need to calculate the style loss & it will calculated between style image &  generated image
            loss_s = 0

            for g_f, s_f in zip(g_feats, s_feats):
                # calc_mean_std(g_f) :- Computes the channel‑wise mean and standard deviation of the feature map g_f.mThese statistics capture the style information (color distribution, texture patterns). Returns (g_mean, g_std).
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)

                # mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std) :- Compares the generated image’s statistics with the style image’s statistics.
                # If they match, the generated image has successfully adopted the style. MSE ensures the generated mean and variance are close to the style’s mean and variance.
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            
            
            # * args.style_weight :- Multiplies the style loss by a weight specified in the command‑line arguments (--style_weight).
            # This weight controls how strongly the style influences the final output.
            loss_s = loss_s * args.style_weight
            # If args.style_weight is large → the stylized image will emphasize textures, colors, and patterns of the style image more strongly.
            # If args.style_weight is small → the stylized image will preserve more of the content structure and be less stylized.

            loss = loss_c + loss_s   # this is the total loss

            # Now we will do back propagation to train this decoder i.e to update its weights using loss functions
            # optimizer.zero_grad() :- Clears (resets) all previously stored gradients in the model parameters.
            # PyTorch accumulates gradients by default, so if you don’t reset them, they’ll keep adding up across iterations. This ensures each training step starts fresh.
            optimizer.zero_grad()
            loss.backward()    # Performs backpropagation: computes the gradient of the loss with respect to all model parameters.
            optimizer.step()    # Updates the model’s parameters using the optimizer (e.g., Adam, SGD).

            # Here we are updating the progress bar’s description so we can see the current losses while training.
            # progress_bar.set_description(...) :- tqdm progress bars let you attach a custom description string that appears alongside the bar. This is useful for showing dynamic info (like losses) during training.
            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            # loss.item() :- Converts the PyTorch tensor loss into a regular Python float.
            # Represents the total loss (content + style, weighted) for the current batch.
            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()


        # Now we will update the learning rate scheduler and then computing the average losses per epoch
        scheduler.step()

        # This gives the average total loss per batch for the epoch.
        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)


        # Checks whether the current epoch number (plus 1, since epochs are zero‑indexed) is divisible by the logging interval.
        # Example: If log_interval=2, it will log at epochs 2, 4, 6, etc. If log_interval=1, it logs every epoch.
        # tqdm.write(...) :- Prints a message above the progress bar without breaking its formatting.
        # Useful for clean logging when using tqdm.
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')


        # torch.save(decoder.state_dict(), ...) :- Saves the decoder’s parameters (weights) to a file named decoder_<epoch>.pth. This lets you resume training or reuse the trained decoder later.
        # torch.save(optimizer.state_dict(), ...) :- Saves the optimizer state (learning rate, momentum, etc.) to a file named optimizer_<epoch>.pth. Important for resuming training exactly where you left off.
        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            # Temporarily disables gradient tracking (since we’re just generating an output, not training). Makes the operation faster and saves memory.
            # here we can use these content_batch, g etc because In Python, variables defined inside a for loop are not limited to the loop’s scope.
            # here these actually hold the last values assigned during the final iteration of the loop.
            with torch.no_grad():
                # Concatenates three sets of images along the batch dimension:
                # Original content images. And Original style images. And Generated stylized images (g).
                # This way, you can visually compare them side by side.
                # torch.cat concatenates tensors along a specified dimension. Here, dim=0 means concatenation along the batch dimension (the first axis).
                # So instead of stacking images side‑by‑side in width or height, you’re stacking them as if they were part of one bigger batch.
                output = torch.cat([content_batch, style_batch, g], dim=0)
                # [content_batch, style_batch, g] :- 
                # Three tensors are being concatenated:
                # content_batch → the original content images.
                # style_batch → the original style images.
                # g → the generated stylized images.
                # Each of these has shape like [N, 3, H, W] (batch size, channels, height, width).
                # After concatenation, output is a single tensor containing all three sets of images.
                # Example: if each batch has 16 images, the result will have: torch.Size([48, 3, H, W]).  That’s 16 content + 16 style + 16 generated images.
                # Makes it easy to save them together in one grid image using save_image.

                # save_image(output, ...) :- save_image is a PyTorch utility (torchvision.utils.save_image) that saves a batch of image tensors as a single image file. It arranges them into a grid for easy visualization
                # Here this output contains Content images, Style images & Generated stylized images. All stacked together as one big batch.
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)
                # nrow=args.batch_size :- Controls how many images are placed per row in the grid.
                # If your batch size is 16, each row will contain 16 images. This makes the saved image neatly organized.






if __name__ == '__main__':
    main()

# Every Python file has a special built‑in variable called __name__.
# If the file is being run directly (e.g., python train.py), then __name__ is set to "__main__".
# If the file is being imported as a module into another script, then __name__ is set to the module’s name (e.g., "train").

# Why use if __name__ == '__main__':
# It ensures that the code inside runs only when the file is executed directly, not when imported.
# In our case, it calls main(), which starts the whole training process (argument parsing, dataset loading, model setup, training loop, saving checkpoints).
# If someone imports this file (e.g., to reuse VGGEncoder, Decoder, or utility functions), the training won’t auto‑start — only the functions/classes will be available.

