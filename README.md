# 손글씨 인식 프로젝트
이 프로젝트는 손글씨 인식을 위한 프로젝트입니다.

## 목차

- [환경](https://github.com/Choigyuhwi/calculator_package/blob/master/README.md#환경)
- [프로젝트 개요](https://github.com/Choigyuhwi/calculator_package/blob/master/README.md#프로젝트-개요)
- [디렉토리 구조](https://github.com/Choigyuhwi/calculator_package/blob/master/README.md#디렉토리-구조)
- [패키지 설치](https://github.com/Choigyuhwi/calculator_package/blob/master/README.md#패키지-설치)
- [사용법](https://github.com/Choigyuhwi/calculator_package/blob/master/README.md#사용법)

## 환경
- Python 3.10.14
- git version 2.25.1
- numpy 1.26.4
- matplotlib 3.9.2

## 프로젝트 개요
이 프로젝트는 손글씨 인식을 위한 프로젝트입니다.
- Handwriting_recognition.py : 손글씨 인식
- Handwriting_recognition_onehot.py : 원핫 인코딩을 사용한 손글씨 인식
- Handwriting_recognition_regession.py : 회귀 분석을 사용한 손글씨 인식  
- MNIST_Optimizer_Activation.py : 옵티마이저와 활성화 함수를 사용한 손글씨 인식
- MNIST_Optimizer.py : 옵티마이저를 사용한 손글씨 인식

## 디렉토리 구조

```
HanddrawingProject/
├── dataset
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz  
├── Handwriting_recognition.py
├── Handwriting_recognition_onehot.py
├── Handwriting_recognition_regession.py
├── MNIST_Optimizer_Activation.py
├── MNIST_Optimizer.py
├── LICENSE
├── README.md
├── setup.py
│


``` 

## 패키지 설치

```bash
git clone https://github.com/Choigyuhwi/HanddrawingProject.git
cd HanddrawingProject
```

## 사용법

### Handwriting_recognition.py
```bash
python Handwriting_recognition.py
```

### Handwriting_recognition_onehot.py
```bash
python Handwriting_recognition_onehot.py
```

### Handwriting_recognition_regession.py
```bash
python Handwriting_recognition_regession.py
```

### MNIST_Optimizer_Activation.py
```bash
python MNIST_Optimizer_Activation.py
```

### MNIST_Optimizer.py
```bash
python MNIST_Optimizer.py
```
