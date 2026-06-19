# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Project :- StyleXfer – "Adaptive Neural Style Transfer"

# StyleXfer is an AI‑powered project built on Adaptive Instance Normalization (AdaIN) that seamlessly transfers artistic styles onto images. 
# By intelligently aligning feature statistics between content and style, it produces visually striking transformations while preserving the essence of the original image. 
# Designed to be lightweight, efficient, and creative, StyleXfer showcases the fusion of deep learning and digital artistry.
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Here in this file, we will write the code of our frontend of our Webapp actually by using the Flask
# This file is essentially a Flask web application that provides a user interface for running our Neural Style Transfer (NST) model using AdaIN. 




# import os → lets you interact with the operating system, including reading environment variables.
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Here we are importing several useful components from Flask, the Python web framework
# Flask :- The main class used to create your web application.
# e.g app = Flask(__name__) -> This initializes our app.
# render_template :- Renders HTML files stored in the templates/ folder. 
# Lets you pass variables into your HTML using Jinja2 templating.
# request :- Handles incoming HTTP requests (form data, query parameters, JSON, etc.).request is a module or object which is used to access the request of the client
# redirect :- Sends the user to a different route after an action.
# url_for :- Dynamically generates URLs for routes or static files. Safer than hardcoding paths.
# url_for generates the correct URL for a given function or static file. Instead of hardcoding paths like /static/style.css or /home, you ask Flask to build them dynamically. This makes your app more portable and avoids broken links when you change routes or deploy under a subpath.
# url_for(endpoint, **values)
# - endpoint → usually the name of the view function (or "static" for static files).
# - values → extra arguments like filenames or route parameters.
# send_from_directory :- Serves files directly from a folder (like images, downloads, or generated outputs).
# e.g return send_from_directory("uploads", filename)
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Here we are importing FlaskForm from the flask_wtf package, which is an extension that integrates WTForms with Flask.
# FlaskForm :- It’s a base class used to create web forms in Flask applications.
from flask_wtf import FlaskForm


# flask_bootstrap :- It’s a Flask extension that integrates the Bootstrap front‑end framework (HTML, CSS, JS) directly into your Flask app.
# Instead of manually downloading Bootstrap files, you can use this extension to quickly style your templates with Bootstrap’s responsive design system.
# Bootstrap :- The Bootstrap class initializes the extension with your Flask app.
# e.g app = Flask(__name__)
# Bootstrap(app)   # attaches Bootstrap support to our app
# Once initialized, you can use Bootstrap’s CSS and components (buttons, forms, navbars, grids) directly in your Jinja2 templates without extra setup.
from flask_bootstrap import Bootstrap

# secure_filename :- It sanitizes file names before saving uploaded files to your server.
# Why: When users upload files via a Flask app, their filenames might contain unsafe characters (spaces, slashes, special symbols) or even malicious paths.
# secure_filename ensures the filename is safe to use on your filesystem.
from werkzeug.utils import secure_filename
# Removes or replaces unsafe characters. Converts spaces to underscores. Strips directory paths (so someone can’t upload a file called ../../etc/passwd). Keeps only ASCII characters.
# e.g filename = secure_filename("my resume (final).pdf")
# print(filename)  # Output: my_resume_final.pdf


# Here we are importing several form field classes from WTForms, which is the library Flask‑WTF builds on to handle web forms
# FileField :- Represents a file upload input (<input type="file">).
# Used when you want users to upload images, PDFs, or other files.
# SubmitField :- Represents a submit button (<input type="submit">). Triggers form submission.
# FloatField :- Represents a numeric input that accepts floating‑point values.
# Useful for things like percentages, weights, or blending strength (alpha in your style transfer app).
# HiddenField :- Represents a hidden input (<input type="hidden">).
# Stores values that shouldn’t be visible to the user but are needed when processing the form (e.g., IDs, tokens).
from wtforms import FileField, SubmitField, FloatField, HiddenField


# here we are importing the InputRequired validator from WTForms. 
# It’s a form validator used in WTForms/Flask‑WTF. Ensures that the user actually provides input for a field before the form can be submitted.
# Unlike DataRequired, which checks that the data isn’t empty after type conversion, InputRequired specifically checks that the input is present in the form submission itself.
from wtforms.validators import InputRequired

from PIL import Image
# Loads the Image class from the Pillow library (PIL = Python Imaging Library).
# Provides tools to open, manipulate, and convert images.
# e.g Image.open(image_path) → opens an image file.
# .convert('RGB') → ensures the image is in RGB format (3 color channels), which is standard for training.



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


# here we are bringing in Python’s built‑in io module, which provides tools for handling streams of data (like files, text, or bytes) in memory.
import io
# io.StringIO :- Acts like a text file stored in memory.
# io.BytesIO :- Acts like a binary file in memory. Useful for images, audio, or any binary data.


# Import our existing AdaIN code 
# Here we are importing these VGGEncoder & Decoder classes from the models.py file of utils folder
from StyleXfer_NST_code.utils.models import VGGEncoder, Decoder
# Here we are importing these functions from utils.py file
from StyleXfer_NST_code.utils.utils import adaptive_instance_normalization, calc_mean_std




# Here this is web app i.e application which will take request & give some response
app = Flask(__name__)
# Flask(__name__) creates a new Flask web application object.
# The __name__ variable tells Flask where to look for resources (like templates or static files). It helps Flask know the “root path” of your app.
# This app object is the central piece: it handles incoming requests and sends back responses.


app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# SECRET_KEY :-
# Used by Flask to secure sessions and forms.
# Required for features like CSRF protection in Flask‑WTF and flash messages.
# Should be kept secret in production (often stored in environment variables instead of hardcoding).

# UPLOAD_FOLDER :-
# Defines the directory where uploaded files will be saved.
# Here, it points to static/uploads, meaning uploaded images will be stored inside your project’s static/uploads folder.
# You’ll typically use it with os.path.join(app.config['UPLOAD_FOLDER'], filename) when saving files.

# ALLOWED_EXTENSIONS :-
# A Python set listing which file types are permitted for upload.
# In this case: only png, jpg, and jpeg images.


# This initializes the Flask‑Bootstrap extension with your Flask application.
# It automatically injects Bootstrap CSS and JS into your templates, so you can use Bootstrap’s responsive design system (buttons, forms, grids, navbars) without manually linking files.
# Example effect: your Flask‑WTF forms can be rendered with Bootstrap styling using built‑in macros like bootstrap/wtf.html.
Bootstrap(app)


# This creates the folder defined in your app’s config (UPLOAD_FOLDER), which in your case is static/uploads.
# os.makedirs creates directories recursively (so if parent folders don’t exist, they’ll be created too).
# The parameter exist_ok=True prevents errors if the folder already exists — it simply does nothing in that case.
# This ensures your app always has a safe place to store uploaded files before you try saving them.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# app.config
# It’s a dictionary‑like object that stores configuration settings for your Flask application.
# You can think of it as a central place where you define key‑value pairs that control how your app behaves.
# Internally, it’s just a subclass of Python’s dict, so you access values with keys like app.config['SECRET_KEY'].



# It defines a Flask‑WTF form class for our style transfer app
class UploadForm(FlaskForm):    # Creates a custom form by inheriting from FlaskForm. Each attribute inside becomes a field in your web form.
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Creates an instance of your VGGEncoder class.
# Loads pretrained weights from the file vgg_normalised.pth (a VGG model trained on ImageNet, normalized for style transfer).
encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)

# Loads the trained weights for the decoder from the file decoder_final.pth.
# This means your decoder has already been trained to reconstruct images, so you don’t start from scratch.
# After this, the encoder provides features, AdaIN adjusts them, and the decoder rebuilds the final stylized image.
decoder.load_state_dict(torch.load('experiment/final_training/decoder_final.pth', map_location=device))


# Switches the model from training mode to evaluation mode.
# This affects certain layers that behave differently during training vs. inference:
# Dropout layers → disabled (no random dropping of neurons).
# BatchNorm layers → use stored running statistics instead of updating them.
encoder.eval()
decoder.eval()




# That function is a helper to validate uploaded filenames in our Flask app
def allowed_file(filename):
    # '.' in filename :- Checks if the filename contains a dot (.). Ensures the file has an extension (e.g., image.png).
    # filename.rsplit('.', 1)[1].lower() :- Splits the filename into two parts: name and extension.
    # rsplit('.', 1) → splits from the right, only once.
    # "photo.png".rsplit('.', 1) → ["photo", "png"]
    # [1] → takes the extension part ("png").
    # .lower() → makes it lowercase, so "JPG" and "jpg" are treated the same.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']




# This is actually the start of our style transfer function in PyTorch + torchvision.
# content_image: the base image (structure to preserve).
# style_image: the style image (artistic look to apply).
# encoder: pretrained VGG encoder to extract features.
# decoder: trained decoder to reconstruct stylized output.
# alpha: blending strength between content and style.
def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    # Creates a preprocessing pipeline for the content image using torchvision.transforms.
    # transforms.Resize(512) → resizes the image so its shorter side is 512 pixels (standardizing input size).
    # transforms.ToTensor() → converts the image into a PyTorch tensor (shape [C, H, W] with values in [0,1]).
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])


    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    # content_transform(content_image) / style_transform(style_image) :-
    # Applies the preprocessing pipeline you defined earlier (Resize, ToTensor, etc.).
    # Converts the raw image into a PyTorch tensor with shape [C, H, W].
    # .unsqueeze(0) :- Adds a new dimension at index 0.
    # This turns the tensor from [C, H, W] → [1, C, H, W].
    # Why? Because PyTorch models expect a batch dimension, even if you’re only passing one image.
    # So now it’s treated as a batch of size 1.
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)


    # It is the core of the style transfer process — it takes our content and style images, extracts features, blends them using AdaIN, and reconstructs the final stylized image
    # with torch.no_grad() :- Disables gradient tracking (no backpropagation). Saves memory and speeds up inference since you’re only generating images, not training as we already have the trained model.
    # The newer and better alternative to torch.no_grad() in PyTorch is torch.inference_mode(). 
    # It was introduced to make inference faster and more memory‑efficient, and is now the recommended choice for evaluation/inference code. 
    # If we want, we can also use this torch.inference_mode() also here.
    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        # Applies AdaIN (Adaptive Instance Normalization). Aligns the mean and variance of the content features to match those of the style features.
        # This is the mathematical blending of style into content.
        # Here we are passing content_feats & not content_feats[-1] because here this content_feats only contains the fetaure map from the last layer only as we mention is_test=True here
        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        # Blends the AdaIN output with the original content features.
        # alpha controls the strength of the style transfer:
        # alpha = 1.0 → full style.
        # alpha = 0.0 → pure content.
        # Values in between → balanced mix.
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        # Passes the blended features into the trained decoder. Reconstructs the final stylized image in RGB space.
        # Output is the image you see with the artistic style applied.
        stylized_image = decoder(stylized_feats)


    return stylized_image




# This function is a utility to save a PyTorch image tensor as a real image file
# image: a PyTorch tensor representing the image you want to save.
# path: the file path (string) where the image should be saved, e.g. "output/stylized.png".
def save_image(image, path):
    # Moves the tensor from GPU to CPU (since PIL can’t handle CUDA tensors).
    # .clone() makes a copy so you don’t accidentally modify the original tensor.
    image = image.cpu().clone()
    
    # Removes the batch dimension. If the tensor shape is [1, C, H, W], it becomes [C, H, W].
    # This is necessary because ToPILImage expects a single image, not a batch.
    image = image.squeeze(0)
    
    # Ensures all pixel values are between 0 and 1.
    # Prevents invalid values (like negatives or >1) from breaking the image conversion.
    image = image.clamp(0, 1)

    # Converts the tensor into a PIL Image object. Now it’s in a format that can be saved as .png, .jpg, etc.
    image = transforms.ToPILImage()(image)

    # Saves the PIL image to the specified file path. Example: "output/stylized.png".
    image.save(path)






# Now we will define the main route of this app which can take both GET & POST requests
# @app.route('/', methods=['GET', 'POST']) :- This is a Flask route decorator.
# It maps the root URL ('/') to the index() function.
# Accepts both GET (loading the page) and POST (submitting the form) requests.
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None

    # Placeholder for error messages. If something goes wrong (invalid file type, missing upload, etc.), this variable will store the error string to show in the template.
    error = None


    # Checks if the form was submitted (POST) and passed validation (all required fields are filled, CSRF token is valid, etc.).
    # This is a Flask‑WTF helper that combines request.method == 'POST' and form.validate().
    if form.validate_on_submit():
        # Ensures the user actually uploaded a content image file.  
        # form.content.data → the uploaded file object.
        # .filename → the name of the uploaded file
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                # Sanitizes the filename using Werkzeug’s secure_filename.
                # Removes unsafe characters (like spaces, slashes, special symbols) to prevent directory traversal or injection attacks. Example: "my photo.png" → "my_photo.png".
                content_filename = secure_filename(form.content.data.filename)
                
                # Saves the uploaded file into your configured upload folder (static/uploads). Uses os.path.join to build the full safe path.
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                # Stores the filename in the hidden field content_path.
                # Useful for keeping track of the file path across requests.
                form.content_path.data = content_filename
        else:
            # If no new file was uploaded, it falls back to the previously stored filename in the hidden field.
            # This way, the app can reuse the last uploaded content image without requiring the user to re‑upload.
            content_filename = form.content_path.data


        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data


        # Checks that both filenames exist (i.e. the user uploaded both a content image and a style image). If either is missing, this block won’t run.
        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            # Here we are using try except block so that errors can be caught (e.g., invalid file, processing failure). If something goes wrong, the app won’t crash — it will fall into the except block.
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)

                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename

                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

                save_image(stylized_image, result_path)
                
                result_image = result_filename
            except Exception as e:
                error = str(e)
        else:
            if not content_filename and not style_filename:
                error = 'Please upload both content and style images'
            elif not content_filename:
                error = 'Please upload content image'
            elif not style_filename:
                error = 'Please upload style image'
    elif request.method == 'POST':
        if not content_filename and not style_filename:
            error = 'Please upload both content and style images'
        elif not content_filename:
            error = 'Please upload content image'
        elif not style_filename:
            error = 'Please upload style image'


    # Tells Flask to render the index.html file from your templates/ folder. This is the page the user sees in their browser.
    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename, style_image=style_filename, error=error)




# It defines a Flask route for serving uploaded files back to the user
# @app.route('/uploads/<filename>') :- Creates a route like /uploads/cat.png.
# <filename> is a dynamic URL parameter — whatever string appears in place of <filename> gets passed into the function.
# The filename argument will contain the actual file name requested in the URL (e.g., "cat.png").
@app.route('/uploads/<filename>')
def send_image(filename):
    # send_from_directory(app.config['UPLOAD_FOLDER'], filename) :-  A Flask helper that safely serves files from a specific directory.
    # app.config['UPLOAD_FOLDER'] → the folder where uploads are stored (e.g., "static/uploads").
    # filename → the requested file inside that folder.
    # Flask will locate the file and return it as an HTTP response so the browser can display or download it.
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




# Flask route for serving example files from a specific folder
# Creates a route like /examples/demo1.png or /examples/subfolder/sample.jpg.
# <path:filename> is a dynamic parameter — it captures not just a single filename but also subdirectories if present (because of path: instead of just string).
# Example: /examples/styles/starry_night.jpg → filename = "styles/starry_night.jpg".
@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)






if __name__ == '__main__':
    # Imports the run_simple function from Werkzeug (the underlying library Flask uses for its development server).
    # run_simple is a lightweight way to start a WSGI server for development.
    from werkzeug.serving import run_simple

    # Load host and port from environment variables
    host = os.environ.get('FLASK_HOST', 'localhost')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    # Starts the Flask app on host localhost and port 5000.
    # Parameters explained:
    # 'localhost' → binds the server to your local machine only (not accessible externally).
    # 5000 → the port number where the app will be available (http://localhost:5000).
    # app → the Flask application object you defined earlier.
    # use_reloader=True → automatically restarts the server when you change code files (hot reload).
    # use_debugger=True → enables the interactive debugger, so if an error occurs you get detailed debug info in the browser.
    run_simple(host, port, app, use_reloader=True, use_debugger=True)

# Every Python file has a special built‑in variable called __name__.
# If the file is being run directly (e.g., python train.py), then __name__ is set to "__main__".
# If the file is being imported as a module into another script, then __name__ is set to the module’s name (e.g., "train").
# This check ensures the server only starts when you run the file directly, not when it’s imported elsewhere.



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SO now we can run the flask app using :-
# python app.py
# ----------OR--------
# flask --app app run 


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Here we will write this process definition inside the Procfile.txt actually.
# Here it is a process definition in a Procfile (used by Heroku and similar platforms) that tells the platform how to start your web service.
# web: gunicorn --bind :$PORT app:app

# web :- Defines the process type.
# web is special: it means this process will handle HTTP requests. Heroku (and Render, if you use it) expects at least one web process for a web app.

# gunicorn :- A production‑grade WSGI HTTP server for Python apps. It’s faster and more robust than running python app.py directly.
# Commonly used to serve Flask, Django, and FastAPI apps in production.

# --bind :$PORT :- Tells Gunicorn to bind to the port provided by the platform.
# $PORT is an environment variable automatically set by Heroku/Render.
# You don’t hardcode 5000 or 8000 because the platform dynamically assigns a port.

# app:app :- Refers to your Python file and the Flask app object inside it.
# First app → the filename (app.py).
# Second app → the Flask application instance inside that file:
# from flask import Flask
# app = Flask(__name__)
