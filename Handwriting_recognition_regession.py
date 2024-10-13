# %%
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random

# 데이터 로드 함수
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  # 이미지 데이터는 16바이트 오프셋부터 시작
    return data.reshape(-1, 28*28) / 255.0  # 28x28 크기로 정규화

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)  # 레이블 데이터는 8바이트 오프셋부터 시작
    return labels

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)
    
    return images, labels

# 활성화 함수와 손실 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mean_squared_error(y, t):
    #회귀 문제에 맞게 손실 함수를 평균 제곱 오차(MSE)로 변경
    return np.mean((y - t) ** 2)

def to_one_hot(t, num_classes=10):
    return np.eye(num_classes)[t]

# 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 신경망 클래스
class HandWritingNN:
    def __init__(self):
        self.network = self.init_network()

    def init_network(self):
        np.random.seed(42)

        network = {}
        network['W1'] = np.random.randn(784, 50) * 0.01
        network['b1'] = np.zeros(50)
        network['W2'] = np.random.randn(50, 100) * 0.01
        network['b2'] = np.zeros(100)
        network['W3'] = np.random.randn(100, 1) * 0.01  # Output node changed to 1
        network['b3'] = np.zeros(1)  # Output bias adjusted
        
        return network

    def forward(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = a3  # Linear output

        return y, (a1, z1, a2, z2, a3)

    def backward(self, x, t, y, cache):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1, z1, a2, z2, a3 = cache
        batch_size = x.shape[0]

        dy = (y - t.reshape(-1, 1)) / batch_size  # Gradient of the loss w.r.t output

        grads = {}
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
        
        dz2 = np.dot(dy, w3.T)
        da2 = sigmoid(z2) * (1 - sigmoid(z2)) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        
        dz1 = np.dot(da2, w2.T)
        da1 = sigmoid(z1) * (1 - sigmoid(z1)) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        
        return grads

    def sgd(self, grads, learning_rate=0.01):
        for key in self.network.keys():
            self.network[key] -= learning_rate * grads[key]

    def predict(self, x):
        y, _ = self.forward(x)
        return y

# 데이터 경로 설정
path = '/home/myckh527/HanddrawingProject/dataset'  # gz 파일이 저장된 경로로 설정

# MNIST 데이터셋 로드
x_train, t_train = load_mnist(path, 'train')
x_test, t_test = load_mnist(path, 't10k')

# 신경망 초기화
model = HandWritingNN()

# 학습 파라미터 설정
learning_rate = 0.01
batch_size = 100
epochs = 300  # 예를 들어 10번의 epoch 동안 학습

# 학습 루프
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        t_batch = t_train[i:i + batch_size]
        
        # 순전파
        y, cache = model.forward(x_batch)
        
        # 손실 함수 계산
        loss = mean_squared_error(y, t_batch)
        
        # 역전파
        grads = model.backward(x_batch, t_batch, y, cache)
        
        # SGD로 가중치 업데이트
        model.sgd(grads, learning_rate)

    # 매 epoch마다 정확도 출력
    predictions = model.predict(x_test)
    predictions_rounded = np.round(predictions).astype(int).flatten()
    accuracy_cnt = np.sum(predictions_rounded == t_test)

    accuracy = float(accuracy_cnt) / len(x_test)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    
# 임의의 이미지 하나를 선택하여 예측
random_index = random.randint(0, len(x_test) - 1)
random_img = x_test[random_index].reshape(1, -1)
random_label = t_test[random_index]

result= model.predict(random_img)
result_label = np.argmax(result)

plt.imshow(random_img.reshape(28, 28))
plt.title(f"Prediction: {result_label}, Real: {random_label}, Final Accuracy: {accuracy * 100:.2f}%")
plt.show()


# %%
