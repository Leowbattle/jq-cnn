Convolutional Neural Network inference implemented in jq.

Relies on this fq branch (not yet merged): https://github.com/Leowbattle/fq/tree/safetensors

Build a docker image: `docker build -t jq-cnn .`

Train network (requires PyTorch): `python3 mnist.py --save-model --epochs [n]`

Run on random MNIST digit: `docker run --rm jq-cnn`

Example output:

```
MNIST CNN jq Classifier Driver
========================================
Loading MNIST dataset...
Loaded 10000 test images
Selected random image at index 3025
Image shape: torch.Size([1, 28, 28])
Actual label: 4

Selected digit: 4
Image (ASCII representation):
+--------------------------------------------------------+
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                    @@@@@@        ::--                  |
|                  @@@@@@@@        @@@@@@                |
|                  @@@@@@**        @@@@@@::              |
|                ++@@@@@@          @@@@@@..              |
|                @@@@@@@@          @@@@@@                |
|              --@@@@@@--        @@@@@@@@                |
|              --@@@@@@          @@@@@@@@                |
|              **@@@@@@          @@@@@@@@                |
|              @@@@@@@@@@--    ##@@@@@@@@                |
|              @@@@@@@@@@@@@@@@@@@@@@@@@@                |
|              --@@@@@@@@@@@@@@@@@@@@@@@@                |
|                ::@@@@@@@@@@@@@@@@@@@@@@                |
|                      ..**@@@@@@@@@@@@%%                |
|                              @@@@@@@@%%                |
|                              @@@@@@@@%%                |
|                              @@@@@@@@@@                |
|                              @@@@@@@@@@                |
|                              @@@@@@@@@@                |
|                              @@@@@@@@@@                |
|                              ++@@@@@@..                |
|                                                        |
|                                                        |
|                                                        |
+--------------------------------------------------------+

Converted image to JSON format
JSON structure shape: 1x28x28

Running classifier command:
fq -f format/safetensors/testdata/nn.jq --raw-file SAFETENSORS format/safetensors/testdata/mnist_cnn.safetensors
Input JSON length: 16300 characters
Predicted class: 4
Execution time: 1.289 seconds

========================================
RESULTS:
Actual digit:    4
Predicted digit: 4
Execution time:  1.289 seconds
✓ CORRECT prediction!
========================================
```
