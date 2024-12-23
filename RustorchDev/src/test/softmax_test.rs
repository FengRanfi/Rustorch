use super::super::layers::softmax::Softmax;
use ndarray::prelude::*;
//此文件用于测试softmax代码正确性

pub fn test_shape()
{   
    let temp_softmax=Softmax::new(4,5,0.1);
    println!("{:#?}",temp_softmax);
}

pub fn test_get_max()
{
    let w=4;
    let h=5;
    let n=10;
    let output=1;
    let mut temp_softmax=Softmax::new(w,h,0.0);
    let x= Array::zeros((n as usize,w as usize));
    let y= Array::zeros((n as usize,output as usize));
    println!("{:#?}",temp_softmax.loss(&x, &y));
    if let Some((tdw,tdx))=temp_softmax.back()
    {
        println!("{:#?}",tdw);
        println!("{:#?}",tdx);
    }
    else 
    {
        println!("梯度为空");
    }
}

pub fn train_softmax_model()
{
    let lr=1e-4;

}