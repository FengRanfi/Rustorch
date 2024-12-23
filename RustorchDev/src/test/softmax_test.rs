use super::super::layers::softmax::Softmax;
use ndarray::prelude::*;
//此文件用于测试softmax代码正确性

pub fn test_shape()
{   
    let temp_softmax=Softmax::new(4,5);
    println!("{:#?}",temp_softmax);
}

pub fn test_get_max()
{
    let w=4;
    let h=5;
    let n=10;
    let output=10;
    let temp_softmax=Softmax::new(w,h);
    let x= Array::zeros((n as usize,w as usize));
    let y= Array::zeros((n as usize,output as usize));
    temp_softmax.loss(x,y);

}