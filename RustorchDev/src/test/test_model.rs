use ndarray::prelude::*;
use crate::layers::linear::Linear;
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

    let n=2;
    let h1=20;
    //定义线性层
    let mut temp_linear=Linear::new(n,h1,0.0);
    //定义softmax层
    let mut temp_softmax=Softmax::new(h1,8,0.00001);


    let lr=1e-3;
    for i in 0..100
    {
        let tlres=temp_linear.forward(&x_train);
        let mut tly:Array2<f64>;
        let mut tlregloss=0.0;
        match tlres
        {
                Some((y,regloss))=>
                {
                    tly=y;
                    tlregloss=regloss
                }
                None=>{
                    eprintln!("tlres returned None, skipping iteration");
                    continue; // 跳过本次循环
                }
        }
        let tsres=temp_softmax.loss(&tly,&y_train)+tlregloss;
        println!("{} {}",i,tsres);
        let mut tdx:Array2<f64>;
        let mut tdw:Array2<f64>;
        match temp_softmax.back()
        {
            Some((dx,dw))=>
                {
                    tdx = dx;
                    tdw = dw;
                }
            None=>{
                eprintln!("tdx and tdw returned None, skipping iteration");
                continue; // 跳过本次循环
            }
        }
        temp_linear.back(&tdx);

    }

}

