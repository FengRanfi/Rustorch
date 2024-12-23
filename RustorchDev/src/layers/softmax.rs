use ndarray::prelude::*;
use super::func_add;
#[derive(Debug)]
pub struct Softmax
{
    w: Array2<f64>  // 权重
}
impl Softmax
{
    //用于初始化一个softmax对象
    pub fn new(row_num:i32,col_num:i32)->Self
    {
        let w= Array::zeros((row_num as usize,col_num as usize));
        Softmax{
            w
        }
    }
    
    pub fn loss(&self,x: Array2<f64>,y: Array2<f64>)->f64
    {   //x n,width
        let mut res=x.dot(&self.w);
        res[[4,4]]=22.0;
        res[[5,2]]=49.0;
        let res_max=func_add::get_max_axis1(&res); 
        //print!("{:#?}",res_max);
        let res_reduce=res-&res_max;
        print!("{:#?}",res_reduce);
        let res_e=func_add::get_exp1(&res_reduce);
        println!("{:#?}",res_e);
        0.2
    }

    // fn back()
    // {

    // }

}