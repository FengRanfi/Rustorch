use ndarray::prelude::*;
//输入一个矩阵，求出每行最大值
pub fn get_max_axis1(res:&Array2<f64>)->Array2<f64>
    {
        let shape_info=res.dim();
        
        //初始化行列长度
        let row_num;
        let col_num;

        //获取行列长度
        (row_num,col_num)=shape_info;

        //创建返回数据
        let mut max_res:Array2<f64>=Array::zeros((row_num as usize,1));

        //测试代码
        //println!("{:?}",shape_info);
        //println!("{}||||{}",row_num,col_num);
        //println!("{}",res[[1,2]]);

        //找出每一行最大值，依次填入max_res
        for i in 0..row_num
        {
            let mut maxnum=f64::NEG_INFINITY;
            for j in 0..col_num
            {
                
                if maxnum<res[[i,j]]
                {
                    maxnum=res[[i,j]];
                }
            }
            max_res[[i,0]]=maxnum;
        }

        max_res
    }


//输入一个矩阵，求该矩阵每一个元素的e的指数
pub fn get_exp1(res:&Array2<f64>)->Array2<f64>
{
    let row_num;
    let col_num;
    (row_num,col_num)=res.dim();
    let mut res_e=Array::zeros((row_num as usize,col_num as usize));
    for i in 0..row_num
    {
        for j in 0..col_num
        {
            res_e[[i,j]]=res[[i,j]].exp()
        }

    }
    res_e
}