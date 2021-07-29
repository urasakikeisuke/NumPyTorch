# NumPyTorch

## About
This is a neural network library implemented almost exclusively in Numpy. And it has been designed to match the PyTorch interface as closely as possible.

## Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/urasakikeisuke/NumPyTorch/blob/main/example/example.ipynb)

```python
import numpy

from NumPyTorch import nn, optim

class SimpleConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_prob: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_channels: int = in_channels
        self.num_classes: int = num_classes

        self.loss: float = None

        self.conv2d_1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1) # 28 -> 14
        self.bn_1 = nn.BatchNorm2d(num_features=64)
        self.relu_1 = nn.ReLU(inplace=False)

        self.linear_1 = nn.Linear(in_features=256*14*14, out_features=self.num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: numpy.ndarray, training: bool) -> numpy.ndarray:
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.linear_1(x)

        return x

    def backward(self, input: numpy.ndarray, target: numpy.ndarray) -> numpy.ndarray:
        self.loss: numpy.ndarray = self.criterion(input, target)

        dout: numpy.ndarray = 1.

        dout = self.criterion.backward(dout)

        dout = self.linear_1.backward(dout)

        dout = self.relu_1.backward(dout)
        dout = self.bn_1.backward(dout)
        dout = self.conv2d_1.backward(dout)
```

```python
net: AlelexNet = AlelexNet(
            in_channels=1, 
            num_classes=config.NUM_CLASSES, 
            dropout_prob=config.DROPOUT_PROB
        )

optimizer: optim.Adam = optim.Adam(
    net,
    lr=config.LEARNING_RATE,
    betas=config.ADAM_BETAS,
)

for epoch_itr in range(1, config.EPOCHS + 1):
    # Training
    for _ in range(1, config.NUM_TRAIN_MAX_STEPS + 1):
        batch_mask = numpy.random.choice(config.NUM_TRAIN_DATA, config.BATCH_SIZE)
        input: numpy.ndarray = dataset['train_img'][batch_mask]
        gt_label: numpy.ndarray = dataset['train_label'][batch_mask]

        output = net(input, training=True)
        net.backward(output, gt_label)

        print(f'Training loss ===> {net.loss}')

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    # Evaluation
    accuracy: float = 0.
    for _ in range(1, config.NUM_EVAL_MAX_STEPS + 1):
        batch_mask = numpy.random.choice(config.NUM_EVAL_DATA, config.BATCH_SIZE)
        input: numpy.ndarray = dataset['eval_img'][batch_mask]
        gt_label: numpy.ndarray = dataset['eval_label'][batch_mask]

        output = net(input, training=False)

        pred: numpy.ndarray = numpy.argmax(output, axis=1)

        accuracy += numpy.sum(pred == gt_label)
    
    print(f'Evaluation accuracy ===> {accuracy * 100 / float(dataset["eval_img"].shape[0])})
```
