Convolutional Neural Network inference implemented in jq.

Relies on this fq branch (not yet merged): https://github.com/Leowbattle/fq/tree/safetensors

Build a docker image: `docker build -t jq-cnn .`

Train network (requires PyTorch): `python3 mnist.py --save-model --epochs [n]`

Run on random MNIST digit: `docker run --rm jq-cnn`
