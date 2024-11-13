# 데이콘 고려대 Medical AI (MAI)


대회 링크

https://dacon.io/competitions/official/236382/overview/description

## **Host**

- 주최/주관 : 고려대학교 의료원, 고려대학교 의과대학, 고려대학교 BK21 융합중개의과학교육연구단
- 기간 : 2024.10.02 ~ 2024.10.31

# 1.Project Outline

## [Title]

H&E 염색된 조직 이미지로부터 유전자 발현 예측

## **[Competition Overview]**

- H&E 염색 된 조직 이미지를 입력으로 받아, 해당 이미지에서 유전자 발현 데이터를 예측하는 AI 모델을 개발한다.
- 제공된 학습 데이터는 이미지와 유전자 발현 정보가 함께 제공되며, 이를 바탕으로 이미지와 유전자 발현 간의 관계를 모델이 학습해야 한다.
- 평가 단계에서는 유전자 발현 정보가 제공되지 않는 새로운 이미지를 입력으로 받아, 이를 통해 유전자 발현 프로 파일을 예측해야 한다.

## **[Evaluation criteria]**

![evaluation_index](https://github.com/user-attachments/assets/40b32383-8cf5-4296-b38d-9b69ed2943a7)

## [Data]

- Train :
    - 학습용 H&E 염색된 조직 이미지 샘플 6992개
    - TRAIN_0000.png ~ TRAIN_6991.png
- Test :
    - 평가용 H&E 염색된 조직 이미지 샘플 2277
    - TEST_0000.png ~ TEST_2276.png
- Train.csv:
    - ID : 샘플 ID
    - path : H&E 염색된 조직 이미지의 경로
    - AL645608.7 ~ AL592183.1 : 각 유전자의 발현 정보 ( 유전자 총 3467개 존재 )
- test.csv:
    - ID : 샘플 ID
    - path : H&E 염색된 조직 이미지의 경로
- sample_submission.scv :
    - ID : 샘플 ID
    - AL645608.7 ~ AL592183.1 : 예측한 각 유전자의 발현 정보 ( 유전자 총 3467개 존재 )

## 사용한 장비

![gpu](https://github.com/user-attachments/assets/9827b5a0-2303-40e1-bb1a-8c1d40d777db)

# 2. Model

## 시도해본 모델

- Resnet50
- EfficientNetb0

### 하이퍼파라미터 수정

- **Learning rate**: 기존 값 3e-4에서 1e-3으로 수정 후, **epoch**을 증가시켰을 때 성능이 **0.4450**에서 **0.4950**으로 향상됨.

### 모델 변경

- **EfficientNetB0**로 모델을 변경한 후 첫 번째 훈련 결과 **0.4854**로 기존 **ResNet50**보다 좋은 성능을 보이지 않음.

### 데이터셋 변경

- 기존 데이터셋을 train 80% / test 20%**로 분할하여 사용했으나, 이후에는 train 80% / val 10% / test 10%**로 데이터셋을 분할하여 사용.

### Optimizer 변경

- 기존 **ReLU**에서 **SiLU**로 수정하여 모델 성능을 개선.

### 앙상블 방식

- **Weighted Average Ensemble** 방식을 사용하여, **EfficientNetB0** 모델을 다섯 개 생성하고 각 모델의 학습률을 다르게 설정하여, 학습함.
- 각 모델의 예측 성능을 평가한 후, 성능에 따라 가중치를 계산하고 가중 평균 방식으로 최종 예측 성능을 평가함.

### 최종 성능

- **0.4950**에서 **0.5139**로 성능이 향상됨.

### EfficientNetB0으로 모델을 선택한 이유

EfficientNetB0에 비해 ResNet50은 더 복잡한 특징을 학습하는데 유리하며, EfficientNetB0은 이미지가 상대적으로 간단하고 작은 해상도에서 좋은 성능을 보이기 때문에 사용함. 또, 더 적은 파라미터를 가지고 있기 때문에 학습 속도면에서도 줄일 수 있었음.

### 아쉬운 점

뒤 늦게 대회를 참여하여 약 일주일 정도의 시간만 참여하게 된 점.

1. **다양한 모델 시도 부족**:
    - **Vision Transformer**나 **Swin Transformer**와 같은 모델을 사용해 볼 생각은 했지만, 시간 및 장비의 한계로 시도하지 못했음.
    - **EfficientNetB7**과 같은 더 큰 모델을 사용하여 성능을 개선할 수 있었으나, 장비의 한계로 인해 사용하지 못했음.
2. **배치 사이즈 제한**:
    - 장비 성능의 한계로 인해 **배치 사이즈**를 높이는데 한계가 있어, 높은 배치 사이즈를 못한 점이 아쉽다.
3. **epoch 수 한계**:
    - 모델 성능을 더욱 개선하기 위해서 높은 **epoch 수**로 계속 학습을 하고 싶었지만, 장비의 한계로 인해 할 수 없었음.

### 배운 점

- optimizer 중 SiLU에 대해 알게 됨. (주로 EfficientNet과 같은 모델에서 사용하며, Sigmoid 함수와 선형 함수의 결합으로 정의됨.)
    - 기울기 소실 문제 완화
    - ReLU같은 활성화 함수는 급격한 출력 변화가 나타날 수 있지만, SiLU의 경우에는 더 자연스러운 변화를 보여줌
    - 음수 입력 값에서 Leaky ReLU와 마찬가지로 작은 기울기를 제공하기 때문에, 기존 ReLU의 죽은 뉴런 문제 발생을 방지한다.
- EfficientNet에 대해 알게 됨.( 주로 이미치 분류에서 쓰임.)
    - EfficientNet은 효율성과 성능의 균형을 극대화 한 모델로, 기존 CNN이 단순히 모델의 Depth, Width를 증가 시켰다면, EfficientNet은 Compound Scaling을 통해 깊이, 너비, 해상도를 동시에 확장하는 모델이다.
        - Compound Scaling : 모델의 Depth, Width, Resolution을 적절한 비율로 확장하여 모델의 성능 최적화에 도움을 줌.
            - 이에 따라 모델이 (B0~B7)까지 크기별로 나뉘어짐.
    - 기존 CNN기반 모델들보다 적은 파라미터와 계산량을 가진 효율성을 고려한 모델.
 - Weighted Average Ensemble : 여러 모델의 예측 결과를 결합하여, 예측값을 가중 평균을 통해 최종 예측을 생성한다. 이를 통해, 기존 모델에서 앙상블만으로 모델의 성능을 조금 더 좋게 만들 수 있다.
