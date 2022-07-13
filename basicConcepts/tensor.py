# 2022-07-05
# pytorch study 1
# TENSOR
from typing import List, Tuple

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor

'''
텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조입니다. 
PyTorch에서는 텐서를 사용하여 모델의 입력(input)과 출력(output), 그리고 모델의 매개변수들을 부호화(encode)합니다.
텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 NumPy 의 ndarray와 유사합니다. 
실제로 텐서와 NumPy 배열(array)은 종종 동일한 내부(underly) 메모리를 공유할 수 있어 데이터를 복수할 필요가 없습니다. 
(NumPy 변환(Bridge) 참고) 텐서는 또한 (Autograd 장에서 살펴볼) 자동 미분(automatic differentiation)에 최적화되어 있습니다. 
darray에 익숙하다면 Tensor API를 바로 사용할 수 있을 것입니다.
'''

## pycharm 단축키 ##
# alt + enter - 미리 type 을 찍어볼 수 있음
# alt + ctl + l - 코드 정렬

"""
데이터로부터 직접(directly) 생성하기
데이터로부터 직접 텐서를 생성할 수 있습니다. 데이터의 자료형(data type)은 자동으로 유추합니다.
"""

data: List[List[int]] = [[1, 2], [3, 4]]
x_data: Tensor = torch.tensor(data)

"""
NumPy 배열로부터 생성하기
텐서는 NumPy 배열로 생성할 수 있습니다. (그 반대도 가능합니다 - NumPy 변환(Bridge) 참고)
"""

np_array: ndarray = np.array(data)
x_np: Tensor = torch.from_numpy(np_array)

'''
다른 텐서로부터 생성하기:
명시적으로 재정의(override)하지 않는다면, 인자로 주어진 텐서의 속성(모양(shape), 자료형(datatype))을 유지합니다.
'''
x_ones: Tensor = torch.ones_like(x_data)  # x_data 의 속성을 유지
print(f"Ones Tensor: \n {x_ones} \n")

x_rand: Tensor = torch.rand_like(x_data, dtype=torch.float)  # 속성만 덮어쓰고 인자 타입도 바꿀 수 있음
print(f"Random Tensor: \n {x_rand} \n")

"""
무작위(random) 또는 상수(constant) 값을 사용하기:
shape은 텐서의 차원(dimension)을 나타내는 튜플(tuple)로, 아래 함수들에서는 출력 텐서의 차원을 결정합니다.
"""
shape: Tuple[int, int] = (2, 3,)  # 띠용 이렇게 해도 에러가 안남
shape1: Tuple[int, int] = (2, 3)  # 무슨 차이인지 찍어봤는데 그냥 똑같음

rand_tensor: Tensor = torch.rand(shape)
ones_tensor: Tensor = torch.ones(shape)
zeros_tensor: Tensor = torch.zeros(shape)
rand_tensor2: Tensor = torch.rand(shape)
ones_tensor2: Tensor = torch.ones(shape)
zeros_tensor2: Tensor = torch.zeros(shape)

print(f"Random Tensor:\n {rand_tensor} \n")
print(f"Random Tensor2:\n {rand_tensor2} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Ones Tensor2: \n {ones_tensor2} \n")
print(f"zeros_tensor: \n {zeros_tensor} \n")
print(f"zeros_tensor2: \n {zeros_tensor2} \n")

tensor = torch.rand(3, 4)

print(f"Shape of Tensor: {tensor.shape}\n")
print(f"Datatype of Tensor{tensor.dtype}\n")
print(f"Device tensor is stored on {tensor.device}\n")

"""
텐서 연산(Operation)
전치(transposing), 인덱싱(indexing), 슬라이싱(slicing), 수학 계산, 선형 대수, 임의 샘플링(random sampling) 등, 100가지 이상의 텐서 연산들을 *여기 에서 확인할 수 있습니다.
*여기 == https://pytorch.org/docs/stable/torch.html
각 연산들은 (일반적으로 CPU보다 빠른) GPU에서 실행할 수 있습니다. Colab을 사용한다면, Edit > Notebook Settings 에서 GPU를 할당할 수 있습니다.
기본적으로 텐서는 CPU에 생성됩니다. .to 메소드를 사용하면 (GPU의 가용성(availability)을 확인한 뒤) GPU로 텐서를 명시적으로 이동할 수 있습니다.
장치들 간에 큰 텐서들을 복사하는 것은 시간과 메모리 측면에서 비용이 많이든다는 것을 기억하세요!
"""
if torch.cuda.is_available():
    tensor = tensor.to("cuda")  # tensor gpu에서 연산하도록 이동
print(f"GPU available : {torch.cuda.is_available()}\nGPU device name : {torch.cuda.get_device_name()}\n\n")

"""
목록에서 몇몇 연산들을 시도해보세요. NumPy API에 익숙하다면 Tensor API를 사용하는 것은 식은 죽 먹기라는 것을 알게 되실 겁니다.
NumPy식의 표준 인덱싱과 슬라이싱:
"""
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

print("git test")
print("git test")