import numpy

import nn, optim

class SimpleConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int, num_classes: int,
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.num_classes: int = num_classes

    	self.conv2d_1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1)  
        self.bn_1 = nn.BatchNorm2d(num_features=64)
        self.relu_1 = nn.ReLU(inplace=False)
        self.linear_1 = nn.Linear(in_features=256*14*14, out_features=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    # 順伝搬
    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
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

gt_label = 1

net: SimpleConvNet = SimpleConvNet(
        in_channels=1, 
        num_classes=10, 
    )

optimizer: optim.Adam = optim.Adam(
    net,
    lr=1e-3,
)

output = net(input, training=True)
net.backward(output, gt_label)

print(f'Training loss ===> {net.loss}')

optimizer.step()
optimizer.zero_grad(set_to_none=True)
