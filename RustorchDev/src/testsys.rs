//use crate::test::softmax_test::test_shape;
use crate::test::softmax_test::test_get_max;
use crate::test::test_model;

pub fn test_sys()
{
    //检测softmax形状是否正确
    //test_shape();
    //检测获取最大值的函是否正确
    //test_get_max();
    //测试csv数据读取
    test_model::get_train_data();
}