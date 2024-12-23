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

    pub fn forward(&mut self,x: &Array2<f64>)->Array2<f64>
    {
        // x (n,r) w(r,c)    return y (n,c)
        self.cache=Some(x.clone());
        x.dot(&self.w)+&self.b
    }

    pub fn back(&mut self,dy: &Array2<f64>)-> Option<(Array2<f64>, Array2<f64>)>
    {  //dy (n,c)  x (n,r) w(r,c)
        if let Some(x)=&self.cache 
        {
            
            let dw=x.t().dot(dy);
            let dx=dy.dot(&self.w.t());
            self.w=&self.w-&dw;
            Some((dx,dw))
        }
        else 
        {
            println!("线性层梯度空");
            None    
        }
    }
}
