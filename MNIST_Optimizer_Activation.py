#%%
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random

# 데이터 로드 함수
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  
    return data.reshape(-1, 28*28) / 255.0  

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)  
    return labels

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)
    return images, labels

# 활성화 함수들
def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def elu(x, alpha=1.0): return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 역전파용 활성화 함수 미분들
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))
def relu_derivative(x): return np.where(x > 0, 1, 0)
def tanh_derivative(x): return 1 - np.tanh(x) ** 2
def leaky_relu_derivative(x, alpha=0.01): return np.where(x > 0, 1, alpha)
def elu_derivative(x, alpha=1.0): return np.where(x > 0, 1, alpha * np.exp(x))

# 손실 함수
def mean_squared_error(y, t): return np.mean((y - t) ** 2)
def to_one_hot(t, num_classes=10): return np.eye(num_classes)[t]

# 최적화 알고리즘
class Optimizer:
    def __init__(self, optimizer_type='sgd', lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = {}
        self.velocity = {}
        self.t = 0  # Adam에서 사용하는 시간 스텝
        self.optimizer_type = optimizer_type

    def update(self, params, grads):
        if self.optimizer_type == 'sgd':
            for key in params:
                params[key] -= self.lr * grads[key]

        elif self.optimizer_type == 'momentum':
            for key in params:
                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(grads[key])
                self.momentum[key] = self.beta1 * self.momentum[key] - self.lr * grads[key]
                params[key] += self.momentum[key]

        elif self.optimizer_type == 'rmsprop':
            for key in params:
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(grads[key])
                self.velocity[key] = self.beta2 * self.velocity[key] + (1 - self.beta2) * (grads[key] ** 2)
                params[key] -= self.lr * grads[key] / (np.sqrt(self.velocity[key]) + self.epsilon)

        elif self.optimizer_type == 'adam':
            self.t += 1
            for key in params:
                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(grads[key])
                    self.velocity[key] = np.zeros_like(grads[key])

                self.momentum[key] = self.beta1 * self.momentum[key] + (1 - self.beta1) * grads[key]
                self.velocity[key] = self.beta2 * self.velocity[key] + (1 - self.beta2) * (grads[key] ** 2)

                m_hat = self.momentum[key] / (1 - self.beta1 ** self.t)
                v_hat = self.velocity[key] / (1 - self.beta2 ** self.t)

                params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# 신경망 클래스
class HandWritingNN:
    def __init__(self, activation_function='sigmoid', optimizer='sgd', learning_rate=0.01):
        self.network = self.init_network()
        self.activation_function, self.activation_derivative = self.set_activation_function(activation_function)
        self.optimizer = Optimizer(optimizer, lr=learning_rate)

    def init_network(self):
        np.random.seed(42)
        network = {}
        network['W1'] = np.random.randn(784, 50) * 0.01
        network['b1'] = np.zeros(50)
        network['W2'] = np.random.randn(50, 100) * 0.01
        network['b2'] = np.zeros(100)
        network['W3'] = np.random.randn(100, 10) * 0.01  # Output layer for 10 classes
        network['b3'] = np.zeros(10)  
        return network

    def set_activation_function(self, activation_name):
        if activation_name == 'sigmoid':
            return sigmoid, sigmoid_derivative
        elif activation_name == 'relu':
            return relu, relu_derivative
        elif activation_name == 'tanh':
            return tanh, tanh_derivative
        elif activation_name == 'leaky_relu':
            return leaky_relu, leaky_relu_derivative
        elif activation_name == 'elu':
            return elu, elu_derivative

    def forward(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.activation_function(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = self.activation_function(a2)
        a3 = np.dot(z2, w3) + b3
        y = a3  # Final layer output before softmax

        return y, (a1, z1, a2, z2, a3)

    def backward(self, x, t, y, cache):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        a1, z1, a2, z2, a3 = cache
        batch_size = x.shape[0]

        dy = (y - t) / batch_size

        grads = {}
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
        dz2 = np.dot(dy, w3.T)
        da2 = self.activation_derivative(z2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        dz1 = np.dot(da2, w2.T)
        da1 = self.activation_derivative(z1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        return grads

    def update(self, grads):
        self.optimizer.update(self.network, grads)

    def predict(self, x):
        y, _ = self.forward(x)
        return np.argmax(y, axis=1)  # Predict class

# 데이터 경로
path = '/home/myckh527/HanddrawingProject/dataset'  # gz 파일이 저장된 경로로 설정
# MNIST 데이터 로드
x_train, t_train = load_mnist(path, 'train')
x_test, t_test = load_mnist(path, 't10k')

# 원핫 인코딩 레이블
t_train_onehot = to_one_hot(t_train, 10)
t_test_onehot = to_one_hot(t_test, 10)

# 학습 파라미터
batch_size = 100
epochs = 10
lr = 0.01
# 활성화 함수와 옵티마이저 선택
activation_function = 'relu'  # 'relu','sigmoid', 'tanh', 'leaky_relu', 'elu'도 가능
optimizer_type = 'adam'  # 'sgd','adam','momentum', 'rmsprop'도 가능

# 모델 초기화
model = HandWritingNN(activation_function, optimizer=optimizer_type, learning_rate=lr)

# 학습 루프
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        t_batch = t_train_onehot[i:i + batch_size]

        # 순전파
        y, cache = model.forward(x_batch)

        # 역전파 및 가중치 업데이트
        grads = model.backward(x_batch, t_batch, y, cache)
        model.update(grads)

    # 매 epoch마다 정확도 출력
    predictions = model.predict(x_test)
    accuracy = np.mean(predictions == t_test)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")

# 임의의 이미지 하나를 선택하여 예측
random_index = random.randint(0, len(x_test) - 1)
random_img = x_test[random_index].reshape(1, -1)
random_label = t_test[random_index]

result= model.predict(random_img)
result_label = result[0]

plt.imshow(random_img.reshape(28, 28))
plt.title(f"Prediction: {result_label}, Real: {random_label}, Final Accuracy: {accuracy * 100:.2f}%")
plt.show()
# %%
