### PyTorch CNN

* 작성자: 나동빈(Dongbin Na / dongbinna@postech.ac.kr)
* 본 코드는 POSTECH의 **CSED703G 수업** 과제로 작성한 코드입니다.

### MNIST Dataset

* <b>[MNIST 학습/평가 전체 소스코드](/PyTorch_CNN_MNIST_Dataset.ipynb)</b>는 Google Colab을 이용해 실행할 수 있도록 작성했습니다.
* MNIST 예제에 대해서 사용한 하이퍼 파라미터는 다음과 같습니다.
  * epoch = 10
  * learning_rate = 0.01
  * weight_decay = 0.0002
  * momentum = 0.9

#### 1. LeNet for MNIST

<img src="https://user-images.githubusercontent.com/16822641/97537367-0f119780-1a02-11eb-9d35-82ed55c6eeed.png" width="80%">

* 실제 구현 코드 (**모델 구현** 파트)

<pre>
class LeNet(nn.Module):
    # 실제로 가중치가 존재하는 레이어만 객체로 만들기
    def __init__(self):
        super(LeNet, self).__init__()
        # 여기에서 (1 x 28 x 28)
        # 입력 채널: 1, 출력 채널: 20 (커널 20개)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        # 여기에서 (20 x 24 x 24)
        # 풀링 이후에 (20 x 12 x 12)
        # 입력 채널: 20, 출력 채널: 50 (커널 50개)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        # 여기에서 (50 x 8 x 8)
        # 풀링 이후에 (50 x 4 x 4)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))
        x = F.max_pool2d(self.conv2(x), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    # 3차원의 컨볼루션 레이어를 flatten
    def num_flat_features(self, x):
        size = x.size()[1:] # 배치는 제외하고
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
</pre>

* 테스트 정확도(Test accuracy): **98.99%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|99.97%|99.90%|99.33%|99.82%|99.59%|99.67%|99.88%|99.78%|99.56%|99.75%|99.73%|
|테스트(test)|99.49%|99.91%|98.64%|99.21%|98.47%|98.43%|99.06%|99.32%|98.46%|98.71%|**98.99%**|

* **[학습된 모델 다운로드 (LeNet 1.64MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EaLEiaWty0VKnJlfSHZwmkoBRMKuKa99rtR1j0m26l8MjA?download=1)**

#### 2. AlexNet for MNIST

<img src="https://user-images.githubusercontent.com/16822641/97537390-18026900-1a02-11eb-828a-80b156ff28e1.png" width="80%">

* 실제 구현 코드 (**모델 구현** 파트)

<pre>
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 여기에서 (1 x 28 x 28)
            # 입력 채널: 1, 출력 채널: 96 (커널 96개)
            nn.Conv2d(1, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5),
            # 여기에서 (96 x 28 x 28)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 여기에서 (96 x 13 x 13)
            # 입력 채널: 96, 출력 채널: 256 (커널 256개)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5),
            # 여기에서 (256 x 13 x 13)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 여기에서 (256 x 6 x 6)
            # 입력 채널: 256, 출력 채널: 384 (커널 384개)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            # 여기에서 (384 x 6 x 6)
            nn.ReLU(inplace=True),
            # 입력 채널: 384, 출력 채널: 384 (커널 384개)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            # 여기에서 (384 x 6 x 6)
            nn.ReLU(inplace=True),
            # 입력 채널: 384, 출력 채널: 384 (커널 384개)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            # 여기에서 (384 x 6 x 6)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 여기에서 (384 x 2 x 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(384 * 2 * 2, 2304),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2304, 10),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
</pre>

* 테스트 정확도(Test accuracy): **99.29%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|99.93%|99.15%|99.83%|99.72%|98.32%|99.69%|99.39%|99.36%|99.30%|99.66%|99.43%|
|테스트(test)|99.80%|99.21%|99.81%|99.90%|98.37%|99.10%|98.43%|99.12%|99.49%|99.60%|**99.29%**|

* **[학습된 모델 다운로드 (AlexNet 29.4MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EZ4xVGbtpOxCgouOSqdK2MMBzySO8NqNg9uvN7vZPKZm4g?download=1)**

#### 3. (ImageNet Pretrained ResNet) Transfer Learning for MNIST

* 실제 구현 코드 (**모델 구현** 파트)

<pre>
net = torchvision.models.resnet18(pretrained=True)

# 마지막 레이어의 차원을 10차원으로 조절
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 10)
net = net.to(device)
</pre>

* Transfer learning을 위해 torchvision.models에 정의된 ResNet18 아키텍처를 따릅니다.

* 테스트 정확도(Test accuracy): **99.64%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|
|테스트(test)|99.90%|99.91%|99.61%|99.70%|99.29%|99.78%|99.27%|99.71%|99.79%|99.41%|**99.64%**|

* **[학습된 모델 다운로드 (ResNet-18 42.7MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EaZXK45firZBmpetb4Mx0MsBm85RU3lIFAu27D9vsg_u0Q?download=1)**

### CIFAR-10 Dataset

* CIFAR-10 제로 베이스 학습 예제에서 사용한 하이퍼 파라미터는 다음과 같습니다.
  * epoch = 200
  * learning_rate
    * 0.1 for epoch [0, 100)
    * 0.01 for epoch [100, 150)
    * 0.001 for epoch [150, 200)
  * weight_decay = 0.0002
  * momentum = 0.9

* CIFAR-10 Transfer Learning 예제에서 사용한 하이퍼 파라미터는 다음과 같습니다.
  * epoch = 10
  * learning_rate
    * 0.01 for epoch [0, 5)
    * 0.001 for epoch [5, 10)
  * weight_decay = 0.0002
  * momentum = 0.9

#### 1. AlexNet for CIFAR10

* [AlexNet을 활용한 CIFAR-10 학습/평가 소스코드](/PyTorch_CNN_AlexNet_CIFAR10_Dataset.ipynb)는 Google Colab에서 실행할 수 있도록 했습니다.

<img src="https://user-images.githubusercontent.com/16822641/97579665-aba15d00-1a35-11eb-88ce-82c40fa99114.png" width="80%">

* 실제 구현 코드 (**모델 구현** 파트)

<pre>
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 여기에서 (3 x 32 x 32)
            # 입력 채널: 3, 출력 채널: 96 (커널 96개)
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5),
            # 여기에서 (96 x 32 x 32)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 여기에서 (96 x 15 x 15)
            # 입력 채널: 96, 출력 채널: 256 (커널 256개)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5),
            # 여기에서 (256 x 15 x 15)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 여기에서 (256 x 7 x 7)
            # 입력 채널: 256, 출력 채널: 384 (커널 384개)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            # 여기에서 (384 x 7 x 7)
            nn.ReLU(inplace=True),
            # 입력 채널: 384, 출력 채널: 384 (커널 384개)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            # 여기에서 (384 x 7 x 7)
            nn.ReLU(inplace=True),
            # 입력 채널: 384, 출력 채널: 384 (커널 384개)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            # 여기에서 (384 x 7 x 7)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 여기에서 (384 x 3 x 3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(384 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
</pre>

* 테스트 정확도(Test accuracy): **86.14%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|99.20%|99.34%|95.66%|98.38%|96.42%|98.42%|97.48%|99.82%|99.24%|99.62%|98.36%|
|테스트(test)|88.70%|93.60%|73.20%|77.00%|81.50%|81.00%|87.20%|94.80%|92.00%|92.40%|**86.14%**|

* **[학습된 모델 다운로드 (AlexNet 70.0MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EYU8SZo5MzxHlzIjlu4We9QBHpk1LO0I3PgTkv2Gurqcow?download=1)**

#### 2. ResNet for CIFAR10

* [ResNet을 활용한 CIFAR-10 학습/평가 소스코드](/PyTorch_CNN_ResNet_CIFAR10_Dataset.ipynb)는 Google Colab에서 실행할 수 있도록 했습니다.

* 실제 구현 코드 (**모델 구현** 파트)

<pre>
# ResNet을 위한 BasicBlock 클래스 정의
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # 3x3 필터를 사용 (너비와 높이를 줄일 때는 stride 값 조절)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

        # 3x3 필터를 사용 (패딩을 1만큼 주기 때문에 너비와 높이가 동일)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

        self.shortcut = nn.Sequential() # identity인 경우
        if stride != 1: # stride가 1이 아니라면, Identity mapping이 아닌 경우
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # (핵심) skip connection
        out = F.relu(out)
        return out


# ResNet 클래스 정의
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # 16개의 3x3 필터(filter)를 사용
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 첫 레이어를 제외하고는 너비와 높이를 줄이기 위해 stride를 2로 설정
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes # 다음 레이어를 위해 채널 수 변경
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])


def ResNet32():
    return ResNet(BasicBlock, [5, 5, 5])


def ResNet44():
    return ResNet(BasicBlock, [7, 7, 7])


def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])


def ResNet110():
    return ResNet(BasicBlock, [18, 18, 18])
</pre>

* **ResNet20** 테스트 정확도(Test accuracy): **85.63%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|95.26%|97.76%|98.40%|96.72%|96.18%|81.74%|88.90%|99.22%|99.10%|96.86%|95.01%|
|테스트(test)|84.60%|92.80%|87.90%|81.90%|85.80%|67.80%|78.40%|93.30%|94.90%|88.90%|**85.63%**|

* **[학습된 모델 다운로드 (ResNet 1.07MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/ETrdVDcb8g1KpyZ6qjSprGMB19bCkRQt88_Sdr_VFXn3UQ?download=1)**

* **ResNet44** 테스트 정확도(Test accuracy): **86.65%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|98.86%|99.28%|94.84%|99.78%|96.74%|92.08%|97.52%|98.70%|99.38%|98.62%|97.58%|
|테스트(test)|89.40%|94.70%|73.70%|89.00%|85.10%|71.10%|86.70%|90.80%|94.70%|91.30%|**86.65%**|

* **[학습된 모델 다운로드 (ResNet 2.60MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EU1EorZKm19LrlbFdBajRzkBYJgRqRusyIr-gT1T9N9jDw?download=1)**

#### 3. (ImageNet Pretrained ResNet) Transfer Learning for CIFAR10

* [Transfer Learning을 활용한 CIFAR-10 학습/평가 소스코드](/PyTorch_CNN_Transfer_Learning_CIFAR10_Dataset.ipynb)는 Google Colab에서 실행할 수 있도록 했습니다.

* 실제 구현 코드 예시 (**모델 구현** 파트)

<pre>
net = torchvision.models.resnet18(pretrained=True)

# 마지막 레이어의 차원을 10차원으로 조절
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 10)
net = net.to(device)
</pre>

#### 3-1) ResNet18 Transfer Learning

* Transfer learning을 위해 torchvision.models에 정의된 ResNet18 아키텍처를 따릅니다.

* 테스트 정확도(Test accuracy): **95.13%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|99.74%|99.64%|99.66%|99.32%|99.60%|98.46%|99.88%|99.78%|99.88%|99.74%|99.57%|
|테스트(test)|95.80%|96.80%|93.70%|89.90%|95.60%|91.70%|97.90%|96.10%|96.90%|96.90%|**95.13%**|

* **[학습된 모델 다운로드 (ResNet18 42.7MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EYPho0CixJRFtvyfmRBdtAYBwTAz7tibxTgZKy1cJbQhkQ?download=1)**

#### 3-2) ResNet50 Transfer Learning

* Transfer learning을 위해 torchvision.models에 정의된 ResNet50 아키텍처를 따릅니다.

* 테스트 정확도(Test accuracy): **96.63%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|99.92%|100.0%|99.96%|99.96%|100.0%|99.98%|100.0%|100.0%|100.0%|99.96%|99.98%|
|테스트(test)|97.10%|98.40%|95.70%|90.70%|96.90%|95.30%|98.70%|98.20%|98.90%|96.40%|**96.63%**|

* **[학습된 모델 다운로드 (ResNet50 90.0MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EQZEEuIYN35HpSjJHtlx91YB-LdMiFq-pczlwqmTNqV4yg?download=1)**
