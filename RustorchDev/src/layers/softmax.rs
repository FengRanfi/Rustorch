use ndarray::prelude::*;
#[derive(Debug)]
pub struct Softmax
{
    w: Array2<f64>  // 权重
}
impl Softmax
{
    //用于初始化一个softmax对象
    pub fn new(width:i32,height:i32)->Self
    {
        let w= Array::zeros((width as usize,height as usize));
        Softmax{
            w
        }
    }
    
    // fn loss(&self,x: Array2<f64>,y: Array2<f64>)->f64
    // {   
    //     let res1=x.dot(&self.w);
        
    // }

    // fn back()
    // {

    // }

}