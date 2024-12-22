use super::super::layers::softmax::Softmax;
//此文件用于测试softmax代码正确性

pub fn test_shape()
{   
    let temp_softmax=Softmax::new(4,5);
    println!("{:#?}",temp_softmax);
}
