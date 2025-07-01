# Use an official Python image as the base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libzbar0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (optional, if needed for debugging or extensions)
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]