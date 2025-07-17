# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . /app

# Install the project in editable mode
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV PYTHONPATH /app

# Command to run the API
CMD ["python", "run_api.py"]
