from abc import ABC,abstractmethod
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn.functional as F
from numpy import ndarray
from typing import Tuple, Union, Optional
from torch.nn.modules.utils import _pair


def im2col_indices(
    x: Tensor,
    kernel_height: int,
    kernel_width: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
) -> Tensor:
    # language=rst
    """
    im2col is a special case of unfold which is implemented inside of Pytorch.

    :param x: Input image tensor to be reshaped to column-wise format.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Input tensor reshaped to column-wise format.
    """
    return F.unfold(x, (kernel_height, kernel_width), padding=padding, stride=stride)


def col2im_indices(
    cols: Tensor,
    x_shape: Tuple[int, int, int, int],
    kernel_height: int,
    kernel_width: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
) -> Tensor:
    # language=rst
    """
    col2im is a special case of fold which is implemented inside of Pytorch.

    :param cols: Image tensor in column-wise format.
    :param x_shape: Shape of original image tensor.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Image tensor in original image shape.
    """
    return F.fold(
        cols, x_shape, (kernel_height, kernel_width), padding=padding, stride=stride
    )


def get_square_weights(
    weights: Tensor, n_sqrt: int, side: Union[int, Tuple[int, int]]
) -> Tensor:
    # language=rst
    """
    Return a grid of a number of filters ``sqrt ** 2`` with side lengths ``side``.

    :param weights: Two-dimensional tensor of weights for two-dimensional data.
    :param n_sqrt: Square root of no. of filters.
    :param side: Side length(s) of filter.
    :return: Reshaped weights to square matrix of filters.
    """
    if isinstance(side, int):
        side = (side, side)

    square_weights = torch.zeros(side[0] * n_sqrt, side[1] * n_sqrt)
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < weights.size(1):
                break

            x = i * side[0]
            y = (j % n_sqrt) * side[1]
            filter_ = weights[:, n].contiguous().view(*side)
            square_weights[x : x + side[0], y : y + side[1]] = filter_

    return square_weights


def get_square_assignments(assignments: Tensor, n_sqrt: int) -> Tensor:
    # language=rst
    """
    Return a grid of assignments.

    :param assignments: Vector of integers corresponding to class labels.
    :param n_sqrt: Square root of no. of assignments.
    :return: Reshaped square matrix of assignments.
    """
    square_assignments = torch.mul(torch.ones(n_sqrt, n_sqrt), -1.0)
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < assignments.size(0):
                break

            square_assignments[
                i : (i + 1), (j % n_sqrt) : ((j % n_sqrt) + 1)
            ] = assignments[n]

    return square_assignments


def reshape_locally_connected_weights(
    w: Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    locations: Tensor,
    input_sqrt: Union[int, Tuple[int, int]],
) -> Tensor:
    # language=rst
    """
    Get the weights from a locally connected layer and reshape them to be two-dimensional and square.

    :param w: Weights from a locally connected layer.
    :param n_filters: No. of neuron filters.
    :param kernel_size: Side length(s) of convolutional kernel.
    :param conv_size: Side length(s) of convolution population.
    :param locations: Binary mask indicating receptive fields of convolution population neurons.
    :param input_sqrt: Sides length(s) of input neurons.
    :return: Locally connected weights reshaped as a collection of spatially ordered square grids.
    """
    kernel_size = _pair(kernel_size)
    conv_size = _pair(conv_size)
    input_sqrt = _pair(input_sqrt)

    k1, k2 = kernel_size
    c1, c2 = conv_size
    i1, i2 = input_sqrt
    c1sqrt, c2sqrt = int(math.ceil(math.sqrt(c1))), int(math.ceil(math.sqrt(c2)))
    fs = int(math.ceil(math.sqrt(n_filters)))

    w_ = torch.zeros((n_filters * k1, k2 * c1 * c2))

    for n1 in range(c1):
        for n2 in range(c2):
            for feature in range(n_filters):
                n = n1 * c2 + n2
                filter_ = w[
                    locations[:, n],
                    feature * (c1 * c2) + (n // c2sqrt) * c2sqrt + (n % c2sqrt),
                ].view(k1, k2)
                w_[feature * k1 : (feature + 1) * k1, n * k2 : (n + 1) * k2] = filter_

    if c1 == 1 and c2 == 1:
        square = torch.zeros((i1 * fs, i2 * fs))

        for n in range(n_filters):
            square[
                (n // fs) * i1 : ((n // fs) + 1) * i2,
                (n % fs) * i2 : ((n % fs) + 1) * i2,
            ] = w_[n * i1 : (n + 1) * i2]

        return square
    else:
        square = torch.zeros((k1 * fs * c1, k2 * fs * c2))

        for n1 in range(c1):
            for n2 in range(c2):
                for f1 in range(fs):
                    for f2 in range(fs):
                        if f1 * fs + f2 < n_filters:
                            square[
                                k1 * (n1 * fs + f1) : k1 * (n1 * fs + f1 + 1),
                                k2 * (n2 * fs + f2) : k2 * (n2 * fs + f2 + 1),
                            ] = w_[
                                (f1 * fs + f2) * k1 : (f1 * fs + f2 + 1) * k1,
                                (n1 * c2 + n2) * k2 : (n1 * c2 + n2 + 1) * k2,
                            ]

        return square


def reshape_conv2d_weights(weights: torch.Tensor) -> torch.Tensor:
    # language=rst
    """
    Flattens a connection weight matrix of a Conv2dConnection

    :param weights: Weight matrix of Conv2dConnection object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    """
    sqrt1 = int(np.ceil(np.sqrt(weights.size(0))))
    sqrt2 = int(np.ceil(np.sqrt(weights.size(1))))
    height, width = weights.size(2), weights.size(3)
    reshaped = torch.zeros(
        sqrt1 * sqrt2 * weights.size(2), sqrt1 * sqrt2 * weights.size(3)
    )

    for i in range(sqrt1):
        for j in range(sqrt1):
            for k in range(sqrt2):
                for l in range(sqrt2):
                    if i * sqrt1 + j < weights.size(0) and k * sqrt2 + l < weights.size(
                        1
                    ):
                        fltr = weights[i * sqrt1 + j, k * sqrt2 + l].view(height, width)
                        reshaped[
                            i * height
                            + k * height * sqrt1 : (i + 1) * height
                            + k * height * sqrt1,
                            (j % sqrt1) * width
                            + (l % sqrt2) * width * sqrt1 : ((j % sqrt1) + 1) * width
                            + (l % sqrt2) * width * sqrt1,
                        ] = fltr

    return reshaped


def Error2IO_Current(
        datum: Optional[Union[float, Tensor]],
        max_current: float = 0.8,
        base_current: float = 0.15,
        error_max: float = 10,
        # TODO  how to normalize and whether to add P_max parameter
        P_max:float = 10
) -> list:
    """
    将error 以一定规则转化为输入到IO当中的电流信号。（未经过编码的）
    转换前后shape保持不变


    返回一个list，list[1] 中存储 error对应的主动肌肉的电流（给IO细胞的电流输入）
                 list[2] 中存储error对应的拮抗肌肉的电流 （给 IO_Anti细胞的电流输入）
    :param datum: 此时刻的error值 shape 为[n1,n2,n3,....]  (1,n)   n为要输入的神经元个数  目前我们应该输入的shape为 [1,IO.n]
    :param exp_constant: 转化曲线中sigmoid的时间常数
    :param max_current:  转化曲线中电流最大值
    :param base_current: 转化曲线静默时的电流值
    :param base_current: 输出的最大值（此处为气压最大值）
    :return list={ current,current_anti}

    """
    if isinstance(datum, float):
        datum = torch.Tensor([datum])
    if datum.data > 0:
        # TODO 电流值数量级的把控--计算公式是否需要修改
        # 公式是想当然的 但是在数值上比较合理  --lys
        Current = base_current + (max_current - base_current)/(1+math.exp( - 10*datum.data+5))
        Current_Anti = base_current

        #归一化 同时进行Tensor格式的转换
        Current = torch.Tensor([Current / max_current])
        Current_Anti = torch.Tensor([Current_Anti / max_current])

    else:
        # TODO Anti的电流公式
        Current = base_current
        Current_Anti = base_current + (max_current - base_current)/(1+math.exp(  10*datum.data+5))

        # 归一化 同时进行Tensor格式的转换
        Current = torch.Tensor([Current / max_current])
        Current_Anti = torch.Tensor([Current_Anti / max_current])

    # TODO 简化了静息状态的操作
    print("----The result of error to current----")
    print("Current : ")
    print(Current)
    print("Current_Anti : ")
    print(Current_Anti)

    return Current, Current_Anti


 # 可以选择使用的kernel
class Kernel(ABC):
    """
    Abstract base class for STDP kernel.
    """
    def __init__(
        self
    )->None:
        """
        abstract init
        """

    @abstractmethod
    def create_result(self,delta_t:Optional[Union[float, Tensor]])->None:
        """
        abstract method
        """



class v1(Kernel):
    """
    Kernel in thesis
    Now support both float and Tensor input
    """
    def __init__(
        self
    )->None:
        # language=rst
        """
        """
        super().__init__(
        )

    def create_result(self,delta_t:Optional[Union[float, Tensor]])-> None:
        if isinstance(delta_t,float):
            self.result = math.exp(delta_t) - math.exp(4 * delta_t)
        else:
            self.result = torch.exp(delta_t)-torch.exp(4*delta_t)
        super().create_result(delta_t)


def Plot_Kernel(
    K:Kernel,
    xmin:float=-10,
    xmax:float=10,
    resolution:float = 0.01
)->None:
    """
    绘制特定kernel的图像
    """
    plt.figure(1);
    x = []
    y = []
    a = np.linspace(xmin,xmax,int((xmax-xmin)/resolution))
    for i in a:
        x.append(i);
        K.create_result(delta_t=i)
        y.append(K.result)
    plt.plot(x,y)
    plt.show()


if  __name__ == "__main__":
    aaa = v1()
    Plot_Kernel(K=aaa,xmin=-4,xmax=0)

#def kernel_v1(deltat:float)-> float:



#def Plot_kernel()



