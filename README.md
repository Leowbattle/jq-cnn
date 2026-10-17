Convolutional Neural Network inference implemented in jq.

Uses [fq](https://github.com/wader/fq) (v0.18.0 or newer) to decode the
safetensors weights and to run the jq program. The safetensors decoder and the
`--raw-file` option are part of upstream fq.

Build a docker image: `docker build -t jq-cnn .`

Run on a random MNIST digit: `docker run --rm jq-cnn`

Or run locally (requires the dependencies in `requirements.txt` and `fq` in
`PATH`):

```
pip install -r requirements.txt
python3 jq_cnn_driver.py
```

The driver loads `nn.jq` and `mnist_cnn.safetensors` from the directory
containing the script and stores the MNIST data in `./data`. Override these
paths with the `NN_JQ`, `MODEL_PATH` and `MNIST_DIR` environment variables, and
the fq binary with `FQ_BIN`.

Train network (requires PyTorch): `python3 mnist.py --save-model --epochs [n]`

Example output:

```
MNIST CNN jq Classifier Driver
========================================
Loading MNIST dataset...
Data directory: /app/data
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
fq -f /app/nn.jq --raw-file SAFETENSORS /app/mnist_cnn.safetensors
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
