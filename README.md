# StyleXfer: Adaptive Neural Style Transfer

![Platform](https://img.shields.io/badge/Platform-Web-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![Deep%20Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Computer%20Vision](https://img.shields.io/badge/Computer%20Vision-Neural%20Style%20Transfer-purple)
![Model](https://img.shields.io/badge/Model-AdaIN-orange)
![Language](https://img.shields.io/badge/Language-Python%203.x-yellow)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-success)

StyleXfer is a Flask-based web application for **Adaptive Neural Style Transfer** using **AdaIN**. It lets a user upload a content image and a style image, choose the style strength, and generate a new image that preserves the structure of the content image while adopting the color, texture, and artistic statistics of the style image.

The project combines a pretrained VGG encoder, a trained decoder network, PyTorch image processing, and a responsive Bootstrap/Jinja web interface. It also includes the training script used to train the decoder and sample content/style/example images for demonstration.

## Overview

StyleXfer provides an end-to-end neural style transfer workflow:

- upload a content image that provides the subject and structure
- upload a style image that provides artistic texture and color statistics
- control the transfer strength with an `alpha` slider
- run AdaIN-based inference through a Flask backend
- preview uploaded images before submission
- view and download the stylized output
- browse static examples inside the web app

The application is useful as:

- a computer vision portfolio project
- a practical implementation of AdaIN neural style transfer
- a Flask plus PyTorch deployment example
- a training and inference pipeline for encoder-decoder image generation

## Demo

### AdaIN Architecture

![AdaIN Algorithm](StyleXfer_NST_code/adain_algorithm.png)

### Example Inputs And Outputs

The app includes example files in `StyleXfer_NST_code/examples/`:

- `brad_pitt.jpg` as a sample content image
- `sketch.png` as a sample sketch style
- `picasso_seated_nude_hr.jpg` as a sample painting style
- `stylized_brad_pitt.jpg` and `stylized_brad_pitt (1).jpg` as generated outputs

These examples are served through the Flask route:

```python
@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)
```

## Why This Project

Traditional neural style transfer methods can be slow because they optimize a generated image for every new content-style pair. AdaIN makes style transfer much faster by directly aligning feature statistics between content and style representations.

StyleXfer demonstrates this idea in a usable web app:

- VGG extracts visual feature representations
- AdaIN transfers style by matching mean and standard deviation
- the decoder reconstructs a final RGB image
- Flask provides an interactive interface for non-technical users

## Problem Statement

Artists, learners, and computer vision developers often need a simple way to experiment with neural style transfer without running notebooks or command-line scripts every time.

StyleXfer solves this by providing:

- a browser-based upload workflow
- an adjustable style strength control
- trained model inference through Flask
- reusable model and utility code
- a separate training script for improving or retraining the decoder

## What The Project Does

The application supports:

- content & style image upload
- file validation for `png`, `jpg`, and `jpeg`
- AdaIN feature transformation
- trained decoder-based image generation
- output saving into the upload folder
- result display inside the web page
- result download
- static example gallery
- loading configuration from `.env`
- local development server through Flask/Werkzeug
- production startup through Gunicorn

## Key Features

- Flask web app with Jinja templates
- Bootstrap-based responsive UI
- Flask-WTF form handling with CSRF support
- PyTorch VGG encoder for feature extraction
- custom decoder network for image reconstruction
- Adaptive Instance Normalization implementation
- style strength control through `alpha`

## Tech Stack

- Python
- Flask
- Jinja2
- Bootstrap
- Flask-WTF
- WTForms
- Flask-Bootstrap
- PyTorch
- Torchvision
- Pillow
- NumPy
- tqdm
- Gunicorn
- python-dotenv

## Project Structure

```text
StyleXfer/
|-- README.md
|-- requirements.txt
|-- Procfile.txt
|-- .gitignore
|-- .env
|-- StyleXfer_NST_code/
|   |-- app.py
|   |-- train.py
|   |-- vgg_normalised.pth
|   |-- adain_algorithm.png
|   |-- templates/
|   |   |-- index.html
|   |-- utils/
|   |   |-- models.py
|   |   |-- utils.py
|   |-- examples/
|   |-- content_data_examples/
|   |-- style_data_examples/
|   |-- content_dataset/
|   |-- style_dataset/
|   |-- static/
|   |   |-- uploads/
|   |-- experiment/
|       |-- final_training/
|           |-- decoder_final.pth
|           |-- args.txt
```

## Important Files

- `StyleXfer_NST_code/app.py` - Flask application entrypoint, upload handling, inference route, model loading, and image serving routes.
- `StyleXfer_NST_code/train.py` - training driver for the AdaIN decoder.
- `StyleXfer_NST_code/utils/models.py` - VGG encoder and decoder model architecture.
- `StyleXfer_NST_code/utils/utils.py` - dataset loader, transforms, AdaIN function, and feature statistics helpers.
- `StyleXfer_NST_code/templates/index.html` - complete web interface with upload form, previews, examples, footer, CSS, and JavaScript.
- `StyleXfer_NST_code/vgg_normalised.pth` - pretrained normalized VGG weights used by the encoder.
- `StyleXfer_NST_code/experiment/final_training/decoder_final.pth` - trained decoder weights used for inference.
- `StyleXfer_NST_code/examples/` - static demo images shown in the app.
- `StyleXfer_NST_code/static/uploads/` - runtime folder for user uploads and generated outputs.
- `requirements.txt` - Python dependencies.
- `Procfile` - Gunicorn startup command for deployment platforms.
- `.env` - local environment variables. This file should not be pushed to GitHub.

## How It Works

1. The user opens the Flask web app.
2. The user uploads a content image and a style image.
3. The browser previews both selected images with JavaScript.
4. Flask validates the file extensions and saves files into `static/uploads/`.
5. Pillow opens both files and converts them to RGB.
6. Torchvision resizes each image and converts it to tensors.
7. The VGG encoder extracts content and style feature maps.
8. AdaIN normalizes content features and applies style feature statistics.
9. The `alpha` value blends stylized features with original content features.
10. The trained decoder reconstructs the stylized RGB image.
11. The output image is saved as `stylized_<content_filename>`.
12. The result appears in the web page with a download button.

## AdaIN Pipeline

Adaptive Instance Normalization transfers style by aligning channel-wise feature statistics.

At a high level:

1. extract content features from the content image
2. extract style features from the style image
3. calculate per-channel mean and standard deviation for both
4. normalize content features
5. rescale normalized content features with style statistics
6. decode the transformed feature map into an image

The core function is implemented in `StyleXfer_NST_code/utils/utils.py`:

```python
def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)
```

## Model Architecture

### VGG Encoder

The encoder is a pretrained VGG-style network loaded from `vgg_normalised.pth`.

It is used only for feature extraction:

- encoder parameters are frozen
- features are extracted up to `relu4_1`
- shallow layers help measure style
- deeper features preserve content structure

During training, the encoder returns multiple feature maps:

- `h1`
- `h2`
- `h3`
- `h4`

During inference, the app uses the deepest feature map for AdaIN transformation.

### Decoder

The decoder is a trainable network that reconstructs an RGB image from AdaIN-transformed features.

It uses:

- reflection padding
- convolution layers
- ReLU activations
- nearest-neighbor upsampling

The trained inference checkpoint is:

```text
StyleXfer_NST_code/experiment/final_training/decoder_final.pth
```

## Training Pipeline

The training workflow is implemented in `StyleXfer_NST_code/train.py`.

At a high level, training performs:

1. parse command-line arguments
2. load content and style image datasets
3. create PyTorch dataloaders
4. load pretrained VGG encoder
5. initialize decoder
6. optionally resume from saved decoder and optimizer checkpoints
7. extract content and style features
8. apply AdaIN to create target features
9. decode generated image
10. calculate content loss
11. calculate style loss using feature mean and standard deviation
12. update decoder weights with Adam
13. save decoder checkpoints, optimizer checkpoints, and output grids

### Losses

The training script uses:

- content loss between generated features and AdaIN target features
- style loss between generated and style feature statistics
- weighted total loss:

```text
total_loss = content_loss * content_weight + style_loss * style_weight
```

### Final Training Configuration

The saved final training arguments are stored in:

```text
StyleXfer_NST_code/experiment/final_training/args.txt
```

The final recorded configuration includes:

| Setting | Value |
|---------|-------|
| Content dataset | `./content_dataset` |
| Style dataset | `./style_dataset` |
| VGG weights | `vgg_normalised.pth` |
| Experiment | `final_training` |
| Final image size | `512` |
| Content size | `512` |
| Style size | `512` |
| Batch size | `8` |
| Epochs | `200` |
| Learning rate | `0.0001` |
| LR decay | `0.00005` |
| Content weight | `1.0` |
| Style weight | `10` |
| Resume training | `True` |
| Save interval | `20` |

## Training Commands

### Basic Training

From inside `StyleXfer_NST_code/`:

```bash
python train.py --batch_size 16 --epochs 160 --experiment final_training
```

### Resume Training

```bash
python train.py ^
  --batch_size 8 ^
  --epochs 200 ^
  --experiment final_training ^
  --final_size 512 ^
  --style_weight 10 ^
  --resume ^
  --decoder_path experiment/final_training/decoder_160.pth ^
  --optimizer_path experiment/final_training/optimizer_160.pth
```

## Setup Instructions

### 1. Clone The Repository

```bash
git clone https://github.com/your-username/StyleXfer.git
cd StyleXfer
```

### 2. Create A Virtual Environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env`

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key
UPLOAD_FOLDER=static/uploads
FLASK_HOST=localhost
FLASK_PORT=5000
```

Notes:

- `SECRET_KEY` is used by Flask-WTF for CSRF/session protection.
- `UPLOAD_FOLDER` defaults to `static/uploads` if not provided.
- `FLASK_HOST` and `FLASK_PORT` are used when running `python app.py`.
- Do not commit `.env` to GitHub.

### 5. Confirm Required Model Files

The Flask app expects these files to exist:

```text
StyleXfer_NST_code/vgg_normalised.pth
StyleXfer_NST_code/experiment/final_training/decoder_final.pth
```

Without these files, inference will fail during model loading.

### 6. Run The Web App

From the project root:

```bash
cd StyleXfer_NST_code
python app.py
```

Then open:

```text
http://localhost:5000
```

You can also run with Flask:

```bash
cd StyleXfer_NST_code
flask --app app run
```

## Production Deployment

The project includes a Gunicorn command in `Procfile`:

```text
web: gunicorn --bind :$PORT app:app
```

For platforms such as Heroku or Render, the file usually needs to be named exactly:

```text
Procfile
```

## Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Render the upload page |
| `POST` | `/` | Process uploaded content/style images and generate output |
| `GET` | `/uploads/<filename>` | Serve uploaded or generated images |
| `GET` | `/examples/<path:filename>` | Serve static example images |


## Strengths

- Implements the real AdaIN algorithm instead of only using a library wrapper
- Includes both training and inference code
- Uses a frozen pretrained VGG encoder and trainable decoder architecture
- Provides a complete Flask web app around the ML pipeline
- Has a clean user workflow with previews, slider control, output display, and download
- Includes sample examples for quick demonstration
- Supports local development and production-style Gunicorn startup

## Limitations

- Inference speed depends heavily on CPU/GPU availability.
- Large images may take longer to process.
- The app currently saves uploads and outputs to local disk.
- There is no database or user account system.
- Uploaded files are not automatically cleaned up.
- Only image files with `png`, `jpg`, and `jpeg` extensions are accepted.
- Model files must be present locally before running the app.
- The current `Procfile.txt` may need to be renamed to `Procfile` for some deployment platforms.

## Future Improvements

- Add automatic cleanup for old uploaded/generated images
- Add image size limits and better upload validation
- Add drag-and-drop upload UI
- Add before/after comparison slider
- Add multiple predefined style presets
- Add download options for different output sizes
- Add Docker support for easier deployment

## Learning Outcomes

This project demonstrates:

- building a Flask web application for ML inference
- using Flask-WTF for secure file upload forms
- serving generated images through Flask routes
- implementing Adaptive Instance Normalization in PyTorch
- building a VGG encoder and decoder architecture
- training a decoder for neural style transfer
- using feature statistics for style representation
- managing model checkpoints
