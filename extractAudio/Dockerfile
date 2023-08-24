# Use the TensorFlow base image
# to build: docker build -t my_tensorflow_app .
# docker run -it --rm --runtime=nvidia --gpus all -v /home/karolwojtulewicz/code/actionformer_release/data:/app/data -v /home/karolwojtulewicz/code/actionformer_release/extractAudio:/app -w /app my_tensorflow_app
FROM tensorflow/tensorflow:2.8.4-gpu

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run when the container starts
CMD [ "python", "extractVGGishFeatures.py" ]
# run indefinitely
# CMD tail -f /dev/null