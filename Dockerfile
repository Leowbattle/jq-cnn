FROM golang:1.25-bookworm AS fq-builder

# Build a pinned version of fq. Upstream includes the safetensors decoder and
# the --raw-file option that nn.jq relies on.
ARG FQ_VERSION=v0.18.0
RUN git clone --depth 1 --branch "${FQ_VERSION}" https://github.com/wader/fq.git /fq
WORKDIR /fq
RUN go build -o /out/fq .

FROM python:3.12-slim-bookworm

WORKDIR /app

# CPU-only PyTorch build; the remaining dependencies come from PyPI.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Only the fq binary is needed at runtime.
COPY --from=fq-builder /out/fq /usr/local/bin/fq

# Copy project files
COPY jq_cnn_driver.py .
COPY mnist.py .
COPY nn.jq .
COPY mnist_cnn.safetensors .
COPY README.md .

# Pre-download MNIST dataset during build to avoid downloading on every run
RUN python3 -c "\
from torchvision import datasets, transforms; \
transform = transforms.Compose([ \
    transforms.ToTensor(), \
    transforms.Normalize((0.1307,), (0.3081,)) \
]); \
datasets.MNIST(root='data', train=True, download=True, transform=transform); \
datasets.MNIST(root='data', train=False, download=True, transform=transform); \
print('MNIST dataset downloaded successfully') \
"

# Default command
CMD ["python3", "jq_cnn_driver.py"]
