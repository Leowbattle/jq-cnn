Convolutional Neural Network inference implemented in jq.

Relies on this fq branch (not yet merged): https://github.com/Leowbattle/fq/tree/safetensors

Train network (you must have PyTorch installed): `python3 mnist.py --save-model --epochs [n]`
Run on random MNIST digit: `python3 jq_cnn_driver.py`
