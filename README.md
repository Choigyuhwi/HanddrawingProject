# Calculator 패키지
이 패키지는 기본 계산기, 공학용 계산기, 복소수 계산기를 제공합니다.

# 사용법

## 기본 계산기

```python

from calculator import Calculator

calc = Calculator()

print(calc.add(1, 2, 3, precision=2))  # 출력: 6.00
print(calc.subtract(10, 2, 3, return_float=True))  # 출력: 5.0
print(calc.multiply(2, 3, 4))  # 출력: 24
print(calc.divide(100, 2, precision=3))  # 출력: 50.000
```

## 공학용 계산기

```python

from calculator import EngineeringCalculator

eng_calc = EngineeringCalculator()

print(eng_calc.add(1, 2, 3, precision=2))  # 출력: 6.00
print(eng_calc.square_root(16, precision=3))  # 출력: 4.000
print(eng_calc.log(100, precision=4))  # 출력: 2.0000
print(eng_calc.sin(30, angle_unit='degree', precision=4))  # 출력: 0.5000
print(eng_calc.divide(5, 0)) "Division by zero is not allowed"
```

## 복소수 계산기

```python

from calculator import ComplexCalculator

complex_calc = ComplexCalculator()
z1 = 1+2j
z2 = 3+4j
print(complex_calc.add(z1, z2))  # 출력: (4+6j)
print(complex_calc.magnitude(z1))  # 출력: 2.23606797749979
print(complex_calc.argument(z1))  # 출력: 1.1071487177940904
print(complex_calc.to_polar(z1))  # 출력: (2.23606797749979, 1.1071487177940904)
print(complex_calc.to_rectangular(2.23606797749979, 1.1071487177940904))  # 출력: (1.0000000000000002+2j)

## 모듈 구성

- basic.py: 기본 계산기 모듈
- engineering.py: 공학용 계산기 모듈
- complex.py: 복소수 계산기 모듈
- utils.py: 공통으로 사용되는 유틸리티 함수 모듈

