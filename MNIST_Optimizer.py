#%%
# 필요한 라이브러리 import
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 신경망 클래스 정의 (입력 크기, 은닉층 크기, 출력 크기, 활성화 함수, 가중치 초기화 방식을 인자로 받음)
class MnistNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn, weight_init):
        super(MnistNN, self).__init__()  # 부모 클래스(nn.Module)의 생성자 호출
        self.activation_fn = activation_fn  # 활성화 함수 설정
        self.fc1 = nn.Linear(input_size, hidden_size)  # 첫 번째 fully connected layer 정의
        self.fc2 = nn.Linear(hidden_size, output_size)  # 두 번째 fully connected layer 정의
        
        weight_init(self.fc1)  # 가중치 초기화 (첫 번째 레이어)
        weight_init(self.fc2)  # 가중치 초기화 (두 번째 레이어)
        
    def forward(self, x):
        x = self.fc1(x)  # 입력을 첫 번째 레이어로 전달
        x = self.activation_fn(x)  # 활성화 함수를 적용
        x = self.fc2(x)  # 두 번째 레이어로 전달
        return x  # 최종 출력 반환

# 최적화 함수 선택
def get_optimizer(opt_name, model, learning_rate):
    # 최적화 함수 이름에 따라 SGD, Adam, Momentum, RMSprop 중 하나를 반환
    if opt_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)  # SGD
    elif opt_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)  # Adam
    elif opt_name == 'Momentum':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # Momentum을 사용하는 SGD
    elif opt_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)  # RMSprop
    else:
        raise ValueError("optimizer 오류")  # 잘못된 optimizer 오류 처리

# 활성화 함수 선택
def get_activation_fuction(act_name):
    print(act_name)
    # 활성화 함수 이름에 따라 ReLU, Sigmoid, Tanh, LeakyReLU, ELU 중 하나를 반환
    if act_name == 'ReLU':
        return nn.ReLU()
    elif act_name == 'Sigmoid':
        return nn.Sigmoid()
    elif act_name == 'Tanh':
        return nn.Tanh()
    elif act_name == 'LeakyReLU':
        return nn.LeakyReLU()
    elif act_name == 'ELU':
        return nn.ELU()
    else:
        raise ValueError("잘못된 활성화 함수입니다.")  # 잘못된 활성화 함수 오류 처리
    
# Xavier 초기화 방식 정의
def weight_init_x(value):
    if isinstance(value, nn.Linear):
        nn.init.xavier_uniform_(value.weight)  # 가중치를 Xavier 방식으로 초기화

# He 초기화 방식 정의
def weight_init_he(value):
    if isinstance(value, nn.Linear):
        nn.init.kaiming_uniform_(value.weight, nonlinearity='relu')  # 가중치를 He 방식으로 초기화

# Zero 초기화 방식 정의
def weight_init_z(value):
    if isinstance(value, nn.Linear):
        nn.init.zeros_(value.weight)  # 가중치를 0으로 초기화

# Zero 초기화 방식을 사용한 랜덤 초기화 (잘못된 방식)
def weight_init_random(value):
     if isinstance(value, nn.Linear):
        nn.init.zeros_(value.weight)  # 가중치를 0으로 초기화

# 학습 함수
def train(act_name, weight_init, read_optimizer_name):
    input_size = 10  # 입력 크기
    hidden_size = 50  # 은닉층 크기
    output_size = 1  # 출력 크기
    epochs = 10  # 학습할 에폭 수
    learning_rate = 0.01  # 학습률

    optimizer_name = read_optimizer_name  # 사용하려는 최적화 함수 이름
    activation_fuction = get_activation_fuction(act_name)  # 활성화 함수 가져오기
    weight_init_fuction = weight_init  # 가중치 초기화 함수 가져오기
    
    model = MnistNN(input_size, hidden_size, output_size, activation_fuction, weight_init_fuction)  # 모델 생성
    optimizer = get_optimizer(optimizer_name, model, learning_rate)  # 최적화 함수 설정
    criterion = nn.MSELoss()  # 손실 함수로 MSE 사용
    
    losses = []  # 각 에폭의 손실을 저장할 리스트
    
    # 학습 루프
    for epoch in range(epochs):
        optimizer.zero_grad()  # 이전 에폭의 기울기 초기화
        inputs = torch.randn(100, input_size)  # 무작위 입력 데이터 생성
        targets = torch.randn(100, output_size)  # 무작위 타겟 데이터 생성
        outputs = model(inputs)  # 모델을 통해 예측값 계산
        loss = criterion(outputs, targets)  # 예측값과 타겟값의 손실 계산
        loss.backward()  # 손실에 대한 기울기 계산
        optimizer.step()  # 기울기를 통해 모델의 파라미터 업데이트
        
        losses.append(loss.item())  # 각 에폭의 손실 값을 저장
    
    return losses  # 손실 리스트 반환

# 사용할 활성화 함수 목록
activations = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU']
# 사용할 최적화 함수 목록
optimizers = ['SGD', 'Adam', 'Momentum', 'RMSprop']
# 가중치 초기화 방식을 weight_init_x로 고정
weight_init = weight_init_x  
# 체크 변수 (0이면 활성화 함수 테스트, 1이면 최적화 함수 테스트)
check = 0

# check가 1이면 활성화 함수별로 손실 변화량을 그래프로 표시
if check == 0:
    optimizers_name = optimizers[3]  # 고정된 optimizer 사용
    plt.figure(figsize=(12, 8))  # 그래프 크기 설정
    for act_name in activations:
        losses = train(act_name, weight_init, optimizers_name)  # 각 활성화 함수에 대해 학습
        plt.plot(range(len(losses)), losses, label=f"Activation: {act_name}")  # 손실 그래프 그리기

    plt.xlabel('Epoch')  # x축 레이블
    plt.ylabel('Loss')  # y축 레이블
    plt.title(f"Loss Curves for Different optimizer Functions (Optimizer: {optimizers_name})")  # 그래프 제목
    plt.legend(loc='upper right')  # 범례 위치 설정
    plt.show()  # 그래프 표시

# check가 1이면 최적화 함수별로 손실 변화량을 그래프로 표시
else:
    activations_name = activations[0]  # 고정된 활성화 함수 사용
    plt.figure(figsize=(12, 8))  # 그래프 크기 설정
    for optimizers_name in optimizers:
        losses = train(activations_name, weight_init, optimizers_name)  # 각 최적화 함수에 대해 학습
        plt.plot(range(len(losses)), losses, label=f"optimizers: {optimizers_name}")  # 손실 그래프 그리기

    plt.xlabel('Epoch')  # x축 레이블
    plt.ylabel('Loss')  # y축 레이블
    plt.title(f"Loss Curves for Different Activation Functions (activation: {activations_name})")  # 그래프 제목
    plt.legend(loc='upper right')  # 범례 위치 설정
    plt.show()  # 그래프 표시
# %%
