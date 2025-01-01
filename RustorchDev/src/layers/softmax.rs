// 1. 导入必要的模块和依赖。
// 1. Import necessary modules and dependencies.
// - `ndarray::prelude::*`: 提供多维数组（`Array`）和其相关操作。
//   Provides multi-dimensional arrays (`Array`) and related operations.
// - `ndarray_rand::rand_distr::Uniform` 和 `ndarray_rand::RandomExt`: 用于生成随机的权重矩阵。
//   Used to generate random weight matrices.
// - `super::func_add`: 引入外部定义的辅助函数。
//   Introduces external utility functions.

// 2. 定义 Softmax 结构体。
// 2. Define the `Softmax` struct.
// - `w`: 权重矩阵，存储模型参数。
//   Weight matrix, storing model parameters.
// - `row_num` 和 `col_num`: 输入和输出的维度信息。
//   Dimensions of the input and output.
// - `reg`: 正则化强度，用于控制权重的惩罚项。
//   Regularization strength to control weight penalties.
// - `cache`: 可选的缓存，用于存储前向传播中计算的中间结果（包括独热编码的标签、预测得分、输入数据）。
//   Optional cache to store intermediate results from forward propagation (e.g., one-hot labels, predicted scores, input data).

// 3. 实现 Softmax 结构体的主要方法。
// 3. Implement the main methods of the `Softmax` struct.

// - `new`: 初始化一个 Softmax 对象。
//   Initialize a `Softmax` object.
//   - 通过均匀分布（-1.0 到 1.0）随机生成权重矩阵 `w`。
//     Generate the weight matrix `w` randomly using a uniform distribution (-1.0 to 1.0).
//   - 设置初始值，包括输入/输出维度、正则化系数和缓存。
//     Set initial values including input/output dimensions, regularization coefficient, and cache.

// - `loss`: 计算损失值。
//   Compute the loss value.
//   - 输入 `x` 为样本数据，`y` 为对应的标签（行向量形式）。
//     Input `x` is sample data, and `y` is the corresponding label (row vector format).
//   - 使用矩阵乘法计算未归一化的得分 `res`，然后通过减去每行最大值提高数值稳定性。
//     Compute the unnormalized scores `res` using matrix multiplication, and subtract the maximum of each row to improve numerical stability.
//   - 计算 `res` 的指数值，并归一化得到预测概率分布 `res_score`。
//     Compute the exponent of `res` and normalize it to obtain the predicted probability distribution `res_score`.
//   - 根据交叉熵损失公式，计算损失值，并添加正则化项。
//     Calculate the loss value using the cross-entropy loss formula and add the regularization term.
//   - 在前向传播中，将独热编码标签 `y_oh`、预测分布 `res_score` 和输入数据存入缓存。
//     During forward propagation, store one-hot encoded labels `y_oh`, predicted scores `res_score`, and input data in the cache.

// - `back`: 进行反向传播。
//   Perform backpropagation.
//   - 读取缓存，计算梯度。
//     Read the cache and compute gradients.
//   - `dres`: 预测分布与独热编码标签的差异。
//     Difference between the predicted distribution and one-hot encoded labels.
//   - `dw`: 权重梯度，通过输入转置和预测分布计算，并加上正则化项。
//     Weight gradient, calculated using the transpose of the input and predicted distribution, with the regularization term added.
//   - `dx`: 输入梯度，通过预测误差和权重转置计算。
//     Input gradient, calculated using prediction error and the transpose of weights.
//   - 更新权重矩阵 `w`，根据学习率和权重梯度进行优化。
//     Update the weight matrix `w` by optimizing with the learning rate and weight gradient.
//   - 返回输入和权重的梯度。
//     Return the gradients of the input and weights.

// - `printw`: 打印当前的权重矩阵，用于调试和观察模型参数。
//   Print the current weight matrix for debugging and observing model parameters.
use ndarray::prelude::*;
use super::func_add;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;



#[derive(Debug)]
pub struct Softmax
{
    w: Array2<f64>,  // 权重
    row_num:i32,
    col_num:i32,
    reg:f64,
    cache:Option<(Array2<f64>, Array2<f64>, Array2<f64>)>
}
impl Softmax
{
    //用于初始化一个softmax对象
    pub fn new(row_num:i32,col_num:i32,reg:f64)->Self
    {
        // 初始化权重矩阵 w
        let dist = Uniform::new(-1.0, 1.0);
        let w = Array::random((row_num as usize, col_num as usize), dist);
        //let w = Array::zeros((row_num as usize, col_num as usize));
        //println!("{}",w);
        Softmax{
            w:w,
            row_num:row_num,
            col_num:col_num,
            reg:reg,
            cache:None
        }
    }
    

    pub fn loss(&mut self,x: &Array2<f64>,y: &Array2<f64>)->f64
    {   //x n,row_num
        let res=x.dot(&self.w);
        let n;
        let row_num;
        let col_num;
        (n,row_num)=res.dim();
        col_num=self.col_num as usize;
        //res[[4,4]]=22.0;
        //res[[5,2]]=49.0;
        let res_max=func_add::get_max_axis1(&res); 
        //print!("{:#?}",res_max);
        let res_reduce=res-&res_max;
        //print!("{:#?}",res_reduce);
        let res_e=func_add::get_exp1(&res_reduce);
        //println!("{:#?}",res_e);
        let res_e_sum=res_e.sum_axis(Axis(1)).into_shape((n,1)).unwrap();
        //println!("{:#?}",res_e_sum);
        let mut res_score=res_e/res_e_sum;
        //println!("{:#?}",res_score);
        res_score+=1e-15;
        let mut get_y_score=Array::zeros((n,1));
        let mut y_oh=Array::zeros((n,col_num));
        for i in 0..n
        {
            get_y_score[[i,0]]=res_score[[ i,y[[i,0]] as usize ]].ln();
            y_oh[[ i,y[[i,0]] as usize]]=1.0;
        }
        //println!("{:#?}",get_y_score);
        let mut loss=-get_y_score.sum()/(n as f64);
        self.cache=Some((y_oh,res_score,x.clone()));
        loss=loss+(&self.w*&self.w).sum()*self.reg;
        loss
    }


    pub fn back(&mut self)-> Option<(Array2<f64>, Array2<f64>)>
    {
        if let Some((y_oh, res_score, x)) = &self.cache {

            let dres = res_score - y_oh;
            // x (n,r)  w (r,c)   dres (n,c)
            //println!("{:#?}",x);
            //println!("{:#?}",&self.w);
            //println!("{:#?}",dres);
            //let abs_w=func_add::get_abs(&self.w);
            let dw=x.t().dot(res_score)+2.0*&self.reg*&self.w;
            let dx=dres.dot(&self.w.t());
            self.w=&self.w-0.00001*&dw;
            Some((dx,dw))
        } else {
            println!("Cache is not initialized.");
            None
        }
        
    }

    pub fn printw(&self)
    {
        println!("{:#?}",self.w);
    }

}