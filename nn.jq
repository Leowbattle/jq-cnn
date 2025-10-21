def shape:
  if (. | type) == "number" then
    []
  else
    [. | length] + (.[0] | shape)
  end;

def argmax:
  . as $xs |
  $xs | max as $m |
  $xs | index($m);

def dot($a; $b): reduce range($a | length) as $i (0; . + $a[$i] * $b[$i]);

def matrix_sum($a; $b):
  [range($a | length)] | map(
    . as $i |
    [range($a[0] | length)] | map(
      . as $j |
      $a[$i][$j] + $b[$i][$j]
    )
  );

def matrices_add:
  . as $ms |
  reduce $ms[1:][] as $m ($ms[0]; matrix_sum(.; $m));

def matrix_add_scalar($a; $b):
  [range($a | length)] | map(
    . as $i |
    [range($a[0] | length)] | map(
      . as $j |
      $a[$i][$j] + $b
    )
  );

# https://en.wikipedia.org/wiki/Matrix_multiplication
def matmul_ij($a; $b; $i; $j):
  $a | length as $m |
  $a[0] | length as $n |
  reduce range($n) as $k (0; . + $a[$i][$k] * $b[$k][$j]);

def matmul($a; $b):
  $a | length as $m |
  $a[0] | length as $n |
  $b[0] | length as $p |
  if $n != ($b | length) then
    error("Cannot multiply \($m)x\($n) matrix by \($b | length)x\($p) matrix")
  else
    [range($m)] | map(
      . as $i |
      [range($p)] | map(
        . as $j |
        matmul_ij($a; $b; $i; $j)
      )
    )
  end;

# Matrix-vector product
# Matrices are 2D, so a vector [x1, x2, x3, ...] must be converted
# to the form [[x1], [x2], [x3], ...]
# This probably creates extra garbage/allocations but it's easy to implement for now
def matvec($a; $x):
  matmul($a; $x | map([.])) | flatten;

# https://uk.mathworks.com/help/signal/ref/xcorr2.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
def xcorr($x; $y):
  $x | length as $N |
  $y | length as $M |
  def y_m($m):
    if $m < 0 or $m > $M - 1 then
      0
    else
      $y[$m]
    end;
  [range(-($M - 1); $N)] | map(
    . as $k |
    [range($N)] | map(
      . as $l |
      $x[$l] * y_m($l - $k)
    ) | add
  );

def xcorr2($x; $y):
  $x | length as $k_h | #kernel height
  $x[0] | length as $k_w | # kernel width
  $y | length as $h | # image height
  $y[0] | length as $w | # image width

  def y_ij($i; $j):
    if $i < 0 or $i > $w - 1
    or $j < 0 or $j > $h - 1 then
      0
    else
      $y[$i][$j]
    end;

  # Inner sum
  def ij($i; $j):
    [range($k_h)] | map(
      . as $l |
      [range($k_w)] | map(
        . as $m |
        $x[$l][$m] * y_ij($l - $i; $m - $j)
      ) | add
    ) | add;

  [range(-$h + 2; $k_h - 1)] | map(
    . as $i |
    [range(-$w + 2; $k_w - 1)] | map(
      . as $j |
      ij($i; $j)
    )
  );

# https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
def _Linear($A; $b):
  [matvec($A; .), $b] | transpose | map(add);

# https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
def _Conv2d($A; $b):
  . as $input |
  $A | length as $C_out |
  $A[0] | length as $C_in |
  
  def out($C_outj):
    matrix_add_scalar([range($C_in)] | map(
      . as $k |
      xcorr2($input[$k]; $A[$C_outj][$k])
    ) | matrices_add; $b[$C_outj]);
  
  [range($C_out)] | map(
    . as $C_outj |
    out($C_outj)
  );

def map_tensor(f):
  if (. | type) == "number" then
    . | f
  else
    . | map(map_tensor(f))
  end;

# Activation Functions
def ReLU: map_tensor(if . < 0 then 0 else . end);
def Sigmoid: map_tensor(1 / (1 + (-. | exp)));
def Tanh: map_tensor(. | tanh);

def softmax:
  . | map(exp) as $e |
  $e | add as $sum |
  $e | map(. / $sum);

def log_softmax: softmax | map(log);

. as $input |
$SAFETENSORS | safetensors | .tensors as $tensors |

def Linear($name): _Linear($tensors["\($name).weight"].data; $tensors["\($name).bias"].data);
def Conv2d($name): _Conv2d($tensors["\($name).weight"].data; $tensors["\($name).bias"].data);

# https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
def MaxPool2d($kernel_size):
  . | map(
    . as $img |
    $img | length as $h |
    $img[0] | length as $w |
    [range(0; $h; $kernel_size)] | map(
      . as $i |
      [range(0; $w; $kernel_size)] | map(
        . as $j |
        $img[$i:$i + $kernel_size] | map(.[$j:$j + $kernel_size] | max) | max
      )
    )
  );

$input
| MaxPool2d(2)
| Conv2d("conv1")
| ReLU
| MaxPool2d(2)
| flatten
| Linear("fc1")
| log_softmax
| argmax

# go run . -f format/safetensors/testdata/nn.jq --raw-file MLP format/safetensors/testdata/simple_mlp.safetensors --null-input