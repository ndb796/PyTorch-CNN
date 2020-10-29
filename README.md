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

* 테스트 정확도(Test accuracy): **98.99%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|99.97%|99.90%|99.33%|99.82%|99.59%|99.67%|99.88%|99.78%|99.56%|99.75%|99.73%|
|테스트(test)|99.49%|99.91%|98.64%|99.21%|98.47%|98.43%|99.06%|99.32%|98.46%|98.71%|**98.99%**|

* **[학습된 모델 다운로드 (LeNet 1.64MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EaLEiaWty0VKnJlfSHZwmkoBRMKuKa99rtR1j0m26l8MjA?download=1)**

#### 2. AlexNet for MNIST

<img src="https://user-images.githubusercontent.com/16822641/97537390-18026900-1a02-11eb-828a-80b156ff28e1.png" width="80%">

* 테스트 정확도(Test accuracy): **99.29%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|99.93%|99.15%|99.83%|99.72%|98.32%|99.69%|99.39%|99.36%|99.30%|99.66%|99.43%|
|테스트(test)|99.80%|99.21%|99.81%|99.90%|98.37%|99.10%|98.43%|99.12%|99.49%|99.60%|**99.29%**|

* **[학습된 모델 다운로드 (AlexNet 29.4MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EZ4xVGbtpOxCgouOSqdK2MMBzySO8NqNg9uvN7vZPKZm4g?download=1)**

#### 3. (ImageNet Pretrained ResNet) Transfer Learning for MNIST

* Transfer learning을 위해 torchvision.models에 정의된 ResNet18 아키텍처를 따릅니다.

* 테스트 정확도(Test accuracy): **99.64%**

* 클래스별 정확도 분석

|Average precision|0|1|2|3|4|5|6|7|8|9|Total|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
|학습(train)|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|100.0%|
|테스트(test)|99.90%|99.91%|99.61%|99.70%|99.29%|99.78%|99.27%|99.71%|99.79%|99.41%|**99.64%**|

* **[학습된 모델 다운로드 (ResNet-18 42.7MB)](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EaZXK45firZBmpetb4Mx0MsBm85RU3lIFAu27D9vsg_u0Q?download=1)**
