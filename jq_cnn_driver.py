#!/usr/bin/env python3
# AI generated

"""
MNIST CNN Driver for jq classifier

This program:
1. Loads the MNIST dataset using torch
2. Selects a random digit from the dataset
3. Prints the 28x28 image to the terminal in ASCII art
4. Runs the jq CNN classifier (nn.jq) with fq, piping the image as JSON on stdin

Paths default to the files next to this script. They can be overridden with the
FQ_BIN, NN_JQ, MODEL_PATH, and MNIST_DIR environment variables.
"""

import json
import os
import random
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

APP_DIR = Path(__file__).resolve().parent
DEFAULT_NN_JQ = APP_DIR / "nn.jq"
DEFAULT_MODEL = APP_DIR / "mnist_cnn.safetensors"
DEFAULT_DATA_DIR = APP_DIR / "data"


def _env_path(name, default):
    """Return the path from an environment variable, or a default."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def load_settings():
    """Resolve the fq binary and data paths, allowing environment overrides."""
    return {
        "fq": os.environ.get("FQ_BIN", "fq"),
        "nn_jq": _env_path("NN_JQ", DEFAULT_NN_JQ),
        "model": _env_path("MODEL_PATH", DEFAULT_MODEL),
        "data_dir": _env_path("MNIST_DIR", DEFAULT_DATA_DIR),
    }


def print_image_ascii(image_tensor, actual_label):
    """Print the image as ASCII art to the terminal"""
    # Convert tensor to numpy and squeeze to 2D
    img = image_tensor.squeeze().numpy()

    print(f"\nSelected digit: {actual_label}")
    print("Image (ASCII representation):")
    print("+" + "-" * 28*2 + "+")

    # ASCII characters from dark to light
    chars = " .:-=+*#%@"

    for row in img:
        print("|", end="")
        for pixel in row:
            # Normalize pixel value to 0-9 range for character selection
            # The image is normalized with mean=0.1307, std=0.3081
            # We need to map the normalized values back to a reasonable range
            normalized_pixel = (pixel + 0.4242) / 0.8484  # Rough denormalization
            char_idx = int(np.clip(normalized_pixel * len(chars), 0, len(chars) - 1))
            print(chars[char_idx] * 2, end="")  # Double chars for better aspect ratio
        print("|")

    print("+" + "-" * 28*2 + "+")


def tensor_to_json_list(tensor):
    """Convert a tensor to a nested JSON list format expected by jq"""
    # The tensor should be [1, 28, 28] - we want [[[28x28 values]]]
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        # Convert to the format expected by the jq script: [[[row1], [row2], ...]]
        data = tensor.squeeze(0).tolist()  # Remove batch dimension, convert to list
        return [data]  # Wrap in another list as expected by the jq script
    else:
        raise ValueError(f"Expected tensor shape [1, 28, 28], got {tensor.shape}")


def run_jq_classifier(image_json, settings):
    """Run the jq CNN classifier on the image"""
    nn_jq = settings["nn_jq"]
    model = settings["model"]

    if not nn_jq.is_file():
        print(f"Error: jq program not found: {nn_jq}")
        return (None, 0.0)
    if not model.is_file():
        print(f"Error: safetensors model not found: {model}")
        return (None, 0.0)

    try:
        # Convert the image data to JSON string
        json_input = json.dumps(image_json)

        # Command to run
        cmd = [
            settings["fq"],
            "-f", str(nn_jq),
            "--raw-file", "SAFETENSORS", str(model),
        ]

        print(f"\nRunning classifier command:")
        print(shlex.join(cmd))
        print(f"Input JSON length: {len(json_input)} characters")

        # Measure execution time
        start_time = time.perf_counter()

        # Run the command with the JSON as input
        result = subprocess.run(
            cmd,
            input=json_input,
            text=True,
            capture_output=True,
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if result.returncode == 0:
            predicted_class = result.stdout.strip()
            print(f"Predicted class: {predicted_class}")
            print(f"Execution time: {execution_time:.3f} seconds")
            return (int(predicted_class) if predicted_class.isdigit() else None, execution_time)
        else:
            print(f"Error running classifier:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            print(f"return code: {result.returncode}")
            print(f"Execution time: {execution_time:.3f} seconds")
            return (None, execution_time)

    except FileNotFoundError:
        print(f"Exception running classifier: fq not found ({settings['fq']}).")
        print("Install fq or set FQ_BIN to the fq binary.")
        return (None, 0.0)
    except Exception as e:
        print(f"Exception running classifier: {e}")
        return (None, 0.0)


def main():
    """Main function"""
    settings = load_settings()

    print("MNIST CNN jq Classifier Driver")
    print("=" * 40)

    # Set up the same transform as used in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST test dataset
    print("Loading MNIST dataset...")
    print(f"Data directory: {settings['data_dir']}")
    try:
        dataset = datasets.MNIST(
            root=str(settings["data_dir"]),
            train=False,  # Use test set
            download=True,
            transform=transform
        )
        print(f"Loaded {len(dataset)} test images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet connection for first-time download")
        return

    # Select a random image
    random_idx = random.randint(0, len(dataset) - 1)
    image, actual_label = dataset[random_idx]

    print(f"Selected random image at index {random_idx}")
    print(f"Image shape: {image.shape}")
    print(f"Actual label: {actual_label}")

    # Print the image in ASCII
    print_image_ascii(image, actual_label)

    # Convert to JSON format for jq
    try:
        image_json = tensor_to_json_list(image)
        print(f"\nConverted image to JSON format")
        print(f"JSON structure shape: {len(image_json)}x{len(image_json[0])}x{len(image_json[0][0])}")
    except Exception as e:
        print(f"Error converting image to JSON: {e}")
        return

    # Run the classifier
    predicted_class, execution_time = run_jq_classifier(image_json, settings)

    # Display results
    print("\n" + "=" * 40)
    print("RESULTS:")
    print(f"Actual digit:    {actual_label}")
    print(f"Predicted digit: {predicted_class}")
    print(f"Execution time:  {execution_time:.3f} seconds")

    if predicted_class is not None:
        if predicted_class == actual_label:
            print("✓ CORRECT prediction!")
        else:
            print("✗ INCORRECT prediction")
    else:
        print("? FAILED to get prediction")

    print("=" * 40)


if __name__ == "__main__":
    main()
