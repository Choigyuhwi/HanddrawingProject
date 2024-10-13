#%%
import os
import gzip
import numpy as np
import random
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

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def to_one_hot(t, num_classes=10):
    """
    레이블을 원-핫 인코딩으로 변환하는 함수
    :param t: 정답 레이블 (1차원 배열)
    :param num_classes: 클래스 수 (디폴트는 10)
    :return: 원-핫 인코딩된 2차원 배열
    """
    return np.eye(num_classes)[t]

# 신경망 클래스
class HandWritingNN:
    def __init__(self):
         self.network = self.init_network()
        
    # 신경망 초기화 함수
    def init_network(self):
        np.random.seed(42)

        network = {}
        network['W1'] = np.random.randn(784, 50) * 0.01
        network['b1'] = np.zeros(50)
        network['W2'] = np.random.randn(50, 100) * 0.01
        network['b2'] = np.zeros(100)
        network['W3'] = np.random.randn(100, 10) * 0.01
        network['b3'] = np.zeros(10)
        
        return network

    # 순전파 함수
    def forward(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = softmax(a3)

        return y, (a1, z1, a2, z2, a3)

    # 역전파 함수
    def backward(self, x, t, y, cache):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1, z1, a2, z2, a3 = cache
        batch_size = x.shape[0]
        
        t = to_one_hot(t)
        
        dy = (y - t) / batch_size
        
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
    
    # SGD 업데이트 함수
    def sgd(self, grads, learning_rate=0.01):
        for key in self.network.keys():
            self.network[key] -= learning_rate * grads[key]

    # 예측 함수
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
epochs = 500  # 예를 들어 10번의 epoch 동안 학습

# 학습 루프
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        t_batch = t_train[i:i + batch_size]
        
        # 순전파
        y, cache = model.forward(x_batch)
        
        # 손실 함수 계산
        loss = cross_entropy_error(y, to_one_hot(t_batch))
        
        # 역전파
        grads = model.backward(x_batch, t_batch, y, cache)
        
        # SGD로 가중치 업데이트
        model.sgd(grads, learning_rate)

    # 매 epoch마다 정확도 출력
    accuracy_cnt = 0
    for i in range(len(x_test)):
        y = model.predict(x_test[i:i+1])
        p = np.argmax(y)
        if p == t_test[i]:
            accuracy_cnt += 1

    accuracy = float(accuracy_cnt) / len(x_test)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy * 100:.2f}%, Loss: {loss :.4f}")
    
    
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
