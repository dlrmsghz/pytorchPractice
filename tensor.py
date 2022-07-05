# 2022-07-05
# pytorch study 1
# TENSOR
from typing import List

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

x_rand: Tensor = torch.rand_like(x_data, dtype=torch.float) # 속성만 덮어쓰고 인자 타입도 바꿀 수 있음
print(f"Random Tensor: \n {x_rand} \n")