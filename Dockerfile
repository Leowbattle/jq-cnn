FROM golang:1.24-bullseye AS fq-builder

# Clone and build the specific fq branch with safetensors support
RUN git clone https://github.com/Leowbattle/fq.git /fq
WORKDIR /fq
RUN git checkout safetensors
RUN go mod download
RUN go build -o fq .

# Create final image
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the built fq binary from builder stage
COPY --from=fq-builder /fq /fq

# Set up Python dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY jq_cnn_driver.py .
COPY mnist.py .
COPY nn.jq .
COPY mnist_cnn.safetensors .
COPY README.md .

# Create data directory for MNIST downloads
RUN mkdir -p data

# Pre-download MNIST dataset during build to avoid downloading on every run
RUN python3 -c "\
import torch; \
from torchvision import datasets, transforms; \
transform = transforms.Compose([ \
    transforms.ToTensor(), \
    transforms.Normalize((0.1307,), (0.3081,)) \
]); \
datasets.MNIST(root='data', train=True, download=True, transform=transform); \
datasets.MNIST(root='data', train=False, download=True, transform=transform); \
print('MNIST dataset downloaded successfully') \
"

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/fq:${PATH}"

# Copy the nn.jq and mnist_cnn.safetensors to the expected location in fq
RUN mkdir -p /fq/format/safetensors/testdata
RUN cp nn.jq /fq/format/safetensors/testdata/
RUN cp mnist_cnn.safetensors /fq/format/safetensors/testdata/

# Modify the jq_cnn_driver.py to use the correct fq working directory and data path
RUN sed -i 's|cwd="/Users/leob/projects/fq"|cwd="/fq"|g' jq_cnn_driver.py && \
    sed -i "s|root='../data'|root='data'|g" jq_cnn_driver.py

# Default command
CMD ["python3", "jq_cnn_driver.py"]