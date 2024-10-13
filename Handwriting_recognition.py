#%%
import os
import random
import numpy as np
import gzip
import matplotlib.pyplot as plt

# 데이터 로드 함수
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  # 이미지 데이터는 16바이트 오프셋부터 시작
    return data.reshape(-1, 28*28) / 255.0  # 28x28 크기로 정규화

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)  # 레이블 데이터는 8바이트 오프셋부터 시작
    return labels

# MNIST 데이터셋 로드 함수
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)
    
    return images, labels

# 활성화 함수: Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Leaky ReLU의 미분
def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# 소프트맥스 함수
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 손실 함수 (교차 엔트로피)
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / y.shape[0]

# 원핫 인코딩 함수
def one_hot_encoding(labels, num_classes):
    return np.eye(num_classes)[labels]

# 모델 클래스 정의
class WritingModel:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.network = self.init_network(input_size, hidden_size1, hidden_size2, output_size)
    
    # 가중치와 편향 초기화 (He 초기화)
    def init_network(self, input_size, hidden_size1, hidden_size2, output_size):
        network = {}
        network['w1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)  # He 초기화
        network['b1'] = np.zeros(hidden_size1)
        network['w2'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
        network['b2'] = np.zeros(hidden_size2)
        network['w3'] = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        network['b3'] = np.zeros(output_size)
        return network
    
    # 순전파
    def predict(self, x, keep_prob=1.0):
        w1, w2, w3 = self.network['w1'], self.network['w2'], self.network['w3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        # 첫 번째 은닉층
        a1 = np.dot(x, w1) + b1
        z1 = leaky_relu(a1)
        if keep_prob < 1.0:
            mask1 = np.random.rand(*z1.shape) < keep_prob
            z1 *= mask1  # 드롭아웃 적용
            z1 /= keep_prob  # 드롭아웃 보정
        
        # 두 번째 은닉층
        a2 = np.dot(z1, w2) + b2
        z2 = leaky_relu(a2)
        if keep_prob < 1.0:
            mask2 = np.random.rand(*z2.shape) < keep_prob
            z2 *= mask2  # 드롭아웃 적용
            z2 /= keep_prob  # 드롭아웃 보정
        
        # 출력층
        a3 = np.dot(z2, w3) + b3
        y = softmax(a3)

        return y, z1, z2

    # 역전파 (기울기 계산)
    def backward(self, x, t, y, z1, z2):
        grads = {}
        
        w1, w2, w3 = self.network['w1'], self.network['w2'], self.network['w3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        # 출력층의 역전파
        dy = (y - t) / x.shape[0]
        grads['w3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)

        # 은닉층 2의 역전파
        dz2 = np.dot(dy, w3.T) * leaky_relu_derivative(z2)  # Leaky ReLU 미분
        grads['w2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)

        # 은닉층 1의 역전파
        dz1 = np.dot(dz2, w2.T) * leaky_relu_derivative(z1)  # Leaky ReLU 미분
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads

    # SGD 업데이트 함수
    def sgd_update(self, grads, learning_rate=0.001):  # 학습률을 0.001로 낮춤
        for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3'):
            self.network[key] -= learning_rate * grads[key]
            
    # 학습 함수
    def train(self, x_train, t_train, epochs, batch_size, learning_rate=0.001):  # 학습률을 0.001로 낮춤
        data_size = x_train.shape[0]
        for epoch in range(epochs):
            # 미니 배치 생성
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            
            for i in range(0,data_size, batch_size):
                x_batch = x_train[indices[i:i + batch_size]]
                t_batch = t_train[indices[i:i + batch_size]]

                # 순전파
                y, z1, z2 = self.predict(x_batch)

                # 손실이 NaN이면 학습 중단
                loss = cross_entropy_error(y, t_batch)
                if np.isnan(loss):
                   print("Loss is NaN. Stopping training.")
                   break

                # 역전파를 통해 기울기 계산
                grads = self.backward(x_batch, t_batch, y, z1, z2)

                # SGD 업데이트
                self.sgd_update(grads, learning_rate)

                     # 매 epoch마다 손실 및 정확도 출력
            train_loss = cross_entropy_error(self.predict(x_train)[0], t_train)
            train_accuracy = self.evaluate(x_train, t_train)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")


    def evaluate(self, x_test, t_test, batch_size=64):
        data_size = x_test.shape[0]
        accuracy_cnt = 0

        for i in range(0, data_size, batch_size):
            x_batch = x_test[i:i + batch_size]
            t_batch = t_test[i:i + batch_size]

            # 순전파
            y_batch, _, _ = self.predict(x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == np.argmax(t_batch, axis=1))

        return accuracy_cnt / data_size
   
# MNIST 데이터셋 경로 설정
path = '/home/myckh527/HanddrawingProject/dataset'  # gz 파일이 저장된 경로로 설정
# 데이터 로드
x_train, t_train = load_mnist(path, kind='train')
x_test, t_test = load_mnist(path, kind='t10k')

# 데이터가 정상적으로 로드되었는지 확인
print(f"Training data shape: {x_train.shape}, Training labels shape: {t_train.shape}")
print(f"Test data shape: {x_test.shape}, Test labels shape: {t_test.shape}")

# 원핫 인코딩
t_train = one_hot_encoding(t_train, 10)
t_test = one_hot_encoding(t_test, 10)

# 모델 초기화
model = WritingModel(input_size=784, hidden_size1=128, hidden_size2=64, output_size=10)

# 모델 학습
model.train(x_train, t_train, epochs=500, batch_size=50, learning_rate=0.001)  # 학습률 조정

# 모델 평가
accuracy = model.evaluate(x_test, t_test)

print(f"Final Accuracy: {accuracy * 100:.2f}%")

# 임의의 이미지 하나를 선택하여 예측
random_index = random.randint(0, len(x_test) - 1)
random_img = x_test[random_index].reshape(1, -1)
random_label = np.argmax(t_test[random_index])

result, _, _ = model.predict(random_img)
result_label = np.argmax(result)

plt.imshow(random_img.reshape(28, 28), cmap='gray')
plt.title(f"Prediction: {result_label}, Real: {random_label}, Final Accuracy: {accuracy * 100:.2f}%")
plt.show()

# %%
