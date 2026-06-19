# Here we are building a container that runs our StyleXfer project on Hugging Face Spaces

# This sets the base image: a lightweight version of Python 3.12.
# “slim” means fewer preinstalled packages → smaller, faster container.
FROM python:3.12-slim

# Set working directory
WORKDIR /app
# Defines the working directory inside the container. All subsequent commands run inside /app.

# Installs all Python dependencies listed in requirements.txt.
# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copies everything from your local project folder into /app inside the container. This includes requirements.txt, your backend/ code, etc.
# Copy project files into the container
COPY . .

# Declares that the container will listen on port 7860.
# Hugging Face Spaces always uses port 7860
EXPOSE 7860

# This is the start command when the container runs. It launches our Flask app.
# --host 0.0.0.0 makes it accessible externally.
# --port 7860 matches Hugging Face’s required port.
# Run Flask app
CMD ["python", "-m", "flask", "--app", "StyleXfer_NST_code.app", "run", "--host", "0.0.0.0", "--port", "7860"]

