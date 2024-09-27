# https://fastapi.tiangolo.com/deployment/docker/
# Use the official Python slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies for opencv https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code
COPY ./app /code/app

# Set environment variables
ENV WANDB_API_KEY=""
ENV WANDB_ORG=""
ENV WANDB_PROJECT=""
ENV WANDB_MODEL_NAME=""
ENV WANDB_MODEL_VERSION=""

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
