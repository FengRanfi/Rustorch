// 1. 导入必要的模块和依赖。
// 1. Import necessary modules and dependencies.
// - `ndarray::prelude::*`: 提供多维数组（`Array`）和其相关操作。
//   Provides multi-dimensional arrays (`Array`) and related operations.
// - `ndarray_rand::rand_distr::Uniform` 和 `ndarray_rand::RandomExt`: 用于生成随机的权重矩阵。
//   Used to generate random weight matrices.
// - `super::func_add`: 引入外部定义的辅助函数。
//   Introduces external utility functions.

// 2. 定义 Linear 结构体。
// 2. Define the `Linear` struct.
// - `w`: 权重矩阵，存储线性层的参数。
//   Weight matrix, storing the parameters of the linear layer.
// - `b`: 偏置向量，存储线性层的偏置值。
//   Bias vector, storing the bias values of the linear layer.
// - `row_num` 和 `col_num`: 输入和输出的维度信息。
//   Dimensions of the input and output.
// - `reg`: 正则化强度，用于控制权重的惩罚项。
//   Regularization strength to control weight penalties.
// - `cache`: 可选的缓存，用于存储前向传播中的输入数据。
//   Optional cache to store input data from forward propagation.

// 3. 实现 Linear 结构体的主要方法。
// 3. Implement the main methods of the `Linear` struct.

// - `new`: 初始化一个 Linear 对象。
//   Initialize a `Linear` object.
//   - 通过均匀分布（-1.0 到 1.0）随机生成权重矩阵 `w`。
//     Generate the weight matrix `w` randomly using a uniform distribution (-1.0 to 1.0).
//   - 偏置向量 `b` 初始化为零。
//     Initialize the bias vector `b` to zeros.
//   - 设置初始值，包括输入/输出维度、正则化系数和缓存。
//     Set initial values including input/output dimensions, regularization coefficient, and cache.

// - `forward`: 进行前向传播。
//   Perform forward propagation.
//   - 输入 `x` 为样本数据，形状为 `(n, row_num)`。
//     Input `x` is the sample data with shape `(n, row_num)`.
//   - 计算输出 `y = x.dot(w) + b`，得到形状为 `(n, col_num)`。
//     Compute the output `y = x.dot(w) + b`, resulting in shape `(n, col_num)`.
//   - 计算正则化损失项，并返回输出和正则化损失值。
//     Compute the regularization loss term and return the output along with the regularization loss.

// - `back`: 进行反向传播。
//   Perform backpropagation.
//   - 输入 `dy` 为输出梯度，形状为 `(n, col_num)`。
//     Input `dy` is the gradient of the output with shape `(n, col_num)`.
//   - 通过梯度计算获得：
//     Compute gradients for:
//     - `dw`: 权重梯度，通过输入转置和输出梯度计算，并加上正则化项。
//       Weight gradient, calculated using the transpose of input and output gradient, with the regularization term added.
//     - `dx`: 输入梯度，通过输出梯度和权重转置计算。
//       Input gradient, calculated using output gradient and the transpose of weights.
//     - `db`: 偏置梯度，通过对输出梯度的每行求和计算。
//       Bias gradient, calculated by summing the output gradient across rows.
//   - 更新权重矩阵 `w`，根据学习率和梯度优化参数。
//     Update the weight matrix `w` by optimizing with gradients.
//   - 返回输入梯度、权重梯度和偏置梯度。
//     Return the input gradient, weight gr

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use super::func_add;

pub struct Linear
{
    w: Array2<f64>,
    b: Array2<f64>,
    row_num:i32,
    col_num:i32,
    reg:f64,
    cache:Option<Array2<f64>>
}
impl Linear
{
    //初始化Linear对象
    pub fn new(row_num:i32,col_num:i32,reg:f64)->Self
    {
        let dist = Uniform::new(-1.0, 1.0);
        let w = Array::random((row_num as usize, col_num as usize), dist);
        let b=Array::zeros((row_num as usize,1));
        Linear{
            w:w,
            b:b,
            row_num:row_num,
            col_num:col_num,
            reg:reg,
            cache:None
        }
    }

    pub fn forward(&mut self,x: &Array2<f64>)->Option<(Array2<f64>,f64)>
    {
        // x (n,r) w(r,c)    return y (n,c)
        self.cache=Some(x.clone());
        let mut regloss=(&self.w*&self.w).sum()*&self.reg;
        let y=&x.dot(&self.w)+&self.b;
        Some((y,regloss))
    }

    pub fn back(&mut self,dy: &Array2<f64>)-> Option<(Array2<f64>, Array2<f64>,Array2<f64>)>
    {  //dy (n,c)  x (n,r) w(r,c)
        if let Some(x)=&self.cache 
        {
            
            let mut dw=x.t().dot(dy);
            let a;
            let b;
            (a,b)=dy.dim();
            dw=dw+2.0*&self.reg*&self.w;
            let dx=dy.dot(&self.w.t());
            let db=dy.sum_axis(Axis(1)).into_shape((a,1)).unwrap();
            self.w=&self.w-&dw;
            Some((dx,dw,db))
        }
        else 
        {
            println!("线性层梯度空");
            None    
        }
    }
}
