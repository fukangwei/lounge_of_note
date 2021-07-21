---
title: Keras之池化层
categories: 深度学习
date: 2018-12-30 21:31:39
---
### MaxPooling1D

&emsp;&emsp;该函数用于时序数据的最大池化：<!--more-->

``` python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```

- `pool_size`：整数，最大池化的窗口大小。
- `strides`：整数或者是`None`，作为缩小比例的因数，例如`2`会使得输入张量缩小一半。如果是`None`，那么默认值是`pool_size`。
- `padding`：`valid`或者`same`。

&emsp;&emsp;输入尺寸：尺寸是`(batch_size, steps, features)`的`3D`张量。
&emsp;&emsp;输出尺寸：尺寸是`(batch_size, downsampled_steps, features)`的`3D`张量。

### MaxPooling2D

&emsp;&emsp;该函数用于空域数据的最大池化：

``` python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

- `pool_size`：整数或者`2`个整数元组，代表在两个方向`(垂直方向, 水平方向)`缩小比例的因数。`(2, 2)`会把输入张量的两个维度都缩小一半。如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
- `strides`：整数、整数元组或者是`None`，步长值。如果是`None`，那么默认值是`pool_size`。
- `data_format`：一个字符串，`channels_last`(默认值)或者`channels_first`，代表图像的通道维的位置。`channels_last`代表尺寸是`(batch, height, width, channels)`的输入张量，而`channels_first`代表尺寸是`(batch, channels, height, width)`的输入张量。

&emsp;&emsp;输入尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, rows, cols, channels)`的`4D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, rows, cols)`的`4D`张量。

&emsp;&emsp;输出尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, pooled_rows, pooled_cols, channels)`的`4D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, pooled_rows, pooled_cols)`的`4D`张量。

### MaxPooling3D

&emsp;&emsp;该函数用于`3D`(空域或时空域)数据的最大池化：

``` python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

- `pool_size`：`3`个整数的元组，代表缩小`(维度1, 维度2, 维度3)`比例的因数。`(2, 2, 2)`会把`3D`输入张量的每个维度缩小一半。
- `strides`：`3`个整数的元组或者是`None`，步长值。
- `data_format`：一个字符串，`channels_last`(默认值)或者`channels_first`，代表数据的通道维的位置。`channels_last`代表尺寸是`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`的输入张量，而`channels_first`代表尺寸是`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`的输入张量。

&emsp;&emsp;输入尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`的`5D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`的`5D`张量。

&emsp;&emsp;输出尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`的`5D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`的`5D`张量。

### AveragePooling1D

&emsp;&emsp;该函数用于时序数据的平均池化：

``` python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```

- `pool_size`：整数，平均池化的窗口大小。
- `strides`：整数或者是`None`，作为缩小比例的因数，例如`2`会使得输入张量缩小一半。如果是`None`，那么默认值是`pool_size`。

&emsp;&emsp;输入尺寸：尺寸是`(batch_size, steps, features)`的`3D`张量。
&emsp;&emsp;输出尺寸：尺寸是`(batch_size, downsampled_steps, features)`的`3D`张量。

### AveragePooling2D

&emsp;&emsp;该函数用于空域数据的平均池化：

``` python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

- `pool_size`：整数或者`2`个整数元组，代表在两个方向`(垂直方向, 水平方向)`缩小比例的因数。`(2, 2)`会把输入张量的两个维度都缩小一半。如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
- `strides`：整数、整数元组或者是`None`，步长值。如果是`None`，那么默认值是`pool_size`。
- `data_format`：一个字符串，`channels_last`(默认值)或者`channels_first`，输入张量中的维度顺序。`channels_last`代表尺寸是`(batch, height, width, channels)`的输入张量，而`channels_first`代表尺寸是`(batch, channels, height, width)`的输入张量。

&emsp;&emsp;输入尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, rows, cols, channels)`的`4D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, rows, cols)`的`4D`张量。

&emsp;&emsp;输出尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, pooled_rows, pooled_cols, channels)`的`4D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, pooled_rows, pooled_cols)`的`4D`张量。

### AveragePooling3D

&emsp;&emsp;该函数用于`3D`(空域或者时空域)数据的平均池化：

``` python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

- `pool_size`：`3`个整数的元组，代表缩小`(维度1, 维度2, 维度3)`比例的因数。`(2, 2, 2)`会把`3D`输入张量的每个维度缩小一半。
- `strides`：`3`个整数的元组或者是`None`，步长值。
- `data_format`：一个字符串，`channels_last`(默认值)或者`channels_first`，代表数据的通道维的位置。`channels_last`代表尺寸是`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`的输入张量，而`channels_first`代表尺寸是`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`的输入张量。

&emsp;&emsp;输入尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`的`5D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`的`5D`张量。

&emsp;&emsp;输出尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`的`5D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`的`5D`张量。

### GlobalMaxPooling1D

&emsp;&emsp;该函数用于时序数据的全局最大池化：

``` python
keras.layers.GlobalMaxPooling1D()
```

&emsp;&emsp;输入尺寸：尺寸是`(batch_size, steps, features)`的`3D`张量。
&emsp;&emsp;输出尺寸：尺寸是`(batch_size, features)`的`2D`张量。

### GlobalAveragePooling1D

&emsp;&emsp;该函数用于时序数据的全局平均池化：

``` python
keras.layers.GlobalAveragePooling1D()
```

&emsp;&emsp;输入尺寸：尺寸是`(batch_size, steps, features)`的`3D`张量。
&emsp;&emsp;输出尺寸：尺寸是`(batch_size, features)`的`2D`张量。

### GlobalMaxPooling2D

&emsp;&emsp;该函数用于空域数据的全局最大池化：

``` python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

参数`data_format`是一个字符串，`channels_last`(默认值)或者`channels_first`，代表图像的通道维的位置。`channels_last`代表尺寸是`(batch, height, width, channels)`的输入张量，而`channels_first`代表尺寸是`(batch, channels, height, width)`的输入张量。
&emsp;&emsp;输入尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, rows, cols, channels)`的`4D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, rows, cols)`的`4D`张量。

&emsp;&emsp;输出尺寸：尺寸是`(batch_size, channels)`的`2D`张量。

### GlobalAveragePooling2D

&emsp;&emsp;该函数用于空域数据的全局平均池化：

``` python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

参数`data_format`是一个字符串，`channels_last`(默认值)或者`channels_first`，代表图像的通道维的位置。`channels_last`代表尺寸是`(batch, height, width, channels)`的输入张量，而`channels_first`代表尺寸是`(batch, channels, height, width)`的输入张量。

&emsp;&emsp;输入尺寸：

- 如果`data_format = 'channels_last'`，尺寸是`(batch_size, rows, cols, channels)`的`4D`张量。
- 如果`data_format = 'channels_first'`，尺寸是`(batch_size, channels, rows, cols)`的`4D`张量。

&emsp;&emsp;输出尺寸：尺寸是`(batch_size, channels)`的`2D`张量。

&emsp;&emsp;**补充说明**：`pooling`和`unpooling`对应到神经网络的技术上就是`downsampling`和`unsampling`。