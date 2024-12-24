use ndarray::prelude::*;
use crate::layers::softmax::Softmax;

use super::super::layers::softmax;
pub fn get_train_data()//->Option<(Array2<f64>, Array2<f64>)>
{
    let mut x_train=Array::zeros((100,2));
    let mut y_train=Array::zeros((100,1));
    for i in 0..10
    {
        for j in 0..10
        {
            let loc=10*i+j;
            x_train[[loc as usize,0]]=i as f64;
            x_train[[loc as usize,1]]=j as f64;
            y_train[[loc as usize,0]]=((i+j)/3) as f64;
        }
    }
    //println!("{} {}",x_train[[9999,0]],x_train[[9999,1]]);
    //println!("{}",y_train[[9999,0]]);

    
    let mut temp_softmax=Softmax::new(2,8,0.0);
    let lr=1e-3;
    for i in 0..100
    {
        let tloss=temp_softmax.loss(&x_train, &y_train);
        println!("{} {}",i,tloss);
        if let Some((dx,dw))=temp_softmax.back()
        {
            if i==99
            {
                temp_softmax.printw();
            }
        }
        else 
        {
            println!("梯度为空");  
        }

    }

}

