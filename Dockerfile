# Use the official Python image from the Docker Hub
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /project

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app/app.py"]
