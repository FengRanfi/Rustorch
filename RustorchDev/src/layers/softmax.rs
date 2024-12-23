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
        loss=loss+(&self.w*&self.w).sum()*&self.reg;
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
            let dw=x.t().dot(res_score)+2.0*&self.reg*&self.w;
            let dx=dres.dot(&self.w.t());
            self.w=&self.w-0.001*&dw;
            Some((dw,dx))
        } else {
            println!("Cache is not initialized.");
            None
        }
        
    }

}