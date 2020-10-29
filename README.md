### PyTorch CNN

* 작성자: 나동빈(Dongbin Na / dongbinna@postech.ac.kr)
* 본 코드는 POSTECH의 **CSED703G 수업** 과제로 작성한 코드입니다.

### MNIST Dataset

* <b>[전체 소스코드](/PyTorch_CNN_MNIST_Dataset.ipynb)</b>는 Google Colab을 이용해 실행할 수 있도록 작성했습니다.
* MNIST 예제에 대해서 사용한 하이퍼 파라미터는 다음과 같습니다.
  * epoch = 10
  * learning_rate = 0.01
  * weight_decay = 0.0002
  * momentum = 0.9

#### 1. LeNet for MNIST

<img src="https://user-images.githubusercontent.com/16822641/97537367-0f119780-1a02-11eb-9d35-82ed55c6eeed.png" width="80%">

* 실제 구현 코드 (모델 구현 파트)

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

* 실제 구현 코드 (모델 구현 파트)

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

* 실제 구현 코드 (모델 구현 파트)

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
