# Rustorch
This is a Rust ML libiary which used for the Rust courses final Homework.

---

# Rust AI 框架

这是一个用 Rust 实现的轻量级 AI 框架，旨在易于使用和扩展。通过该框架，您可以快速构建和实验机器学习模型，通过定义不同的层和操作来实现功能。

## 特性
- 利用 Rust 的性能，轻量且快速。
- 模块化设计，每一层封装为一个类。
- 支持神经网络层的前向和反向传播。

## 快速开始

### 安装
在项目的 `Cargo.toml` 中添加以下内容：

```toml
[dependencies]
ndarray = "0.15"
ndarray-rand = "0.15"
rand = "0.8"
```

### 示例用法
以下是使用框架实现简单机器学习应用的方法。

#### 步骤 1：导入框架
```rust
use ndarray::prelude::*;
use your_framework::{Linear, Softmax}; // 将 `your_framework` 替换为实际的 crate 名称
```

#### 步骤 2：定义层
```rust
let mut linear_layer = Linear::new(3, 2, 0.01); // 输入特征 3，输出特征 2，正则化强度 0.01
let mut softmax_layer = Softmax::new(2, 2, 0.01); // 输入特征 2，输出特征 2，正则化强度 0.01
```

#### 步骤 3：前向传播和损失计算
向层提供输入数据以进行前向传播，并计算损失：
```rust
let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 输入数据，形状为 (2, 3)
let y_true = array![[0], [1]]; // 目标标签，形状为 (2, 1)

let y = linear_layer.forward(&x);
let loss = softmax_layer.loss(&y, &y_true);
println!("损失值: {}", loss);
```

#### 步骤 4：反向传播
使用反向传播计算梯度：
```rust
if let Some((dx, dw)) = softmax_layer.back() {
    linear_layer.back(&dx);
} else {
    println!("反向传播失败：未缓存数据。");
}
```

### 示例应用
以下是使用框架训练简单神经网络的完整示例：

```rust
use ndarray::prelude::*;
use your_framework::{Linear, Softmax};

fn main() {
    let mut layer1 = Linear::new(3, 2, 0.01);
    let mut layer2 = Softmax::new(2, 2, 0.01);

    // 模拟输入数据和目标数据
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 输入 (2, 3)
    let target = array![[0], [1]];                   // 目标 (2, 1)

    // 前向传播
    let y1 = layer1.forward(&x);
    let loss = layer2.loss(&y1, &target);

    println!("损失值: {}", loss);

    // 反向传播
    if let Some((dy1, _)) = layer2.back() {
        layer1.back(&dy1);
    }
}
```

## API 参考
### `Linear`
支持前向和反向传播的全连接层。

#### 方法
- `new(row_num: i32, col_num: i32, reg: f64) -> Self`
  - 使用指定的维度和正则化强度初始化一个新的 `Linear` 层。
- `forward(&mut self, x: &Array2<f64>) -> Array2<f64>`
  - 执行前向传播。
- `back(&mut self, dy: &Array2<f64>) -> Option<(Array2<f64>, Array2<f64>)>`
  - 执行反向传播，返回相对于输入和权重的梯度。

### `Softmax`
具有损失计算和反向传播功能的 Softmax 层。

#### 方法
- `new(row_num: i32, col_num: i32, reg: f64) -> Self`
  - 使用指定的维度和正则化强度初始化一个新的 `Softmax` 层。
- `loss(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> f64`
  - 计算损失并缓存用于反向传播的中间结果。
- `back(&mut self) -> Option<(Array2<f64>, Array2<f64>)>`
  - 执行反向传播，返回相对于输入和权重的梯度。

## 许可证
本项目使用 MIT 许可证进行许可。


# Rust AI Framework

This is a minimalistic AI framework implemented in Rust, designed for ease of use and extensibility. With this framework, you can quickly build and experiment with machine learning models by defining different layers and operations.

## Features
- Lightweight and fast, leveraging Rust's performance.
- Modular design with each layer encapsulated as a class.
- Support for forward and backward propagation in neural network layers.

## Getting Started

### Installation
To use this framework in your project, include it in your `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.15"
ndarray-rand = "0.15"
rand = "0.8"
```

### Example Usage
Here's how you can use the framework to implement a simple machine learning application.

#### Step 1: Import the Framework
```rust
use ndarray::prelude::*;
use your_framework::{Linear, Softmax}; // Replace `your_framework` with the actual crate name
```

#### Step 2: Define Layers
```rust
let mut linear_layer = Linear::new(3, 2, 0.01); // 3 input features, 2 output features, regularization strength 0.01
let mut softmax_layer = Softmax::new(2, 2, 0.01); // 2 input features, 2 output features, regularization strength 0.01
```

#### Step 3: Forward and Loss Computation
Provide input data to the layers for forward propagation and compute loss:
```rust
let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // Input data with shape (2, 3)
let y_true = array![[0], [1]]; // Target labels with shape (2, 1)

let y = linear_layer.forward(&x);
let loss = softmax_layer.loss(&y, &y_true);
println!("Loss: {}", loss);
```

#### Step 4: Backward Propagation
Compute gradients using backward propagation:
```rust
if let Some((dx, dw)) = softmax_layer.back() {
    linear_layer.back(&dx);
} else {
    println!("Backward propagation failed: no cached data.");
}
```

### Example Application
Here is a full example of using the framework to train a simple neural network:

```rust
use ndarray::prelude::*;
use your_framework::{Linear, Softmax};

fn main() {
    let mut layer1 = Linear::new(3, 2, 0.01);
    let mut layer2 = Softmax::new(2, 2, 0.01);

    // Dummy input and target data
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // Input (2, 3)
    let target = array![[0], [1]];                   // Target (2, 1)

    // Forward pass
    let y1 = layer1.forward(&x);
    let loss = layer2.loss(&y1, &target);

    println!("Loss: {}", loss);

    // Backward pass
    if let Some((dy1, _)) = layer2.back() {
        layer1.back(&dy1);
    }
}
```

## API Reference
### `Linear`
A fully connected layer that supports forward and backward propagation.

#### Methods
- `new(row_num: i32, col_num: i32, reg: f64) -> Self`
    - Initializes a new `Linear` layer with the specified dimensions and regularization strength.
- `forward(&mut self, x: &Array2<f64>) -> Array2<f64>`
    - Performs the forward pass.
- `back(&mut self, dy: &Array2<f64>) -> Option<(Array2<f64>, Array2<f64>)>`
    - Performs the backward pass, returning gradients with respect to inputs and weights.

### `Softmax`
A softmax layer with loss computation and backward propagation.

#### Methods
- `new(row_num: i32, col_num: i32, reg: f64) -> Self`
    - Initializes a new `Softmax` layer with the specified dimensions and regularization strength.
- `loss(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> f64`
    - Computes the loss and caches intermediate results for backward propagation.
- `back(&mut self) -> Option<(Array2<f64>, Array2<f64>)>`
    - Performs the backward pass, returning gradients with respect to inputs and weights.

## License
This project is licensed under the MIT License.


