import pandas as pd
import tensorflow as tf
import numpy as np
df=pd.read_csv('data1.csv')
a=[]
for i in df:
    #print(i)
    a.append(df['%s' % i])
b=np.array(a).transpose()
print(b.shape[0])
print(b)
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))#定义一个随机矩阵变量，行数为in_size,列数为out_size
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)#定义一个一行，out_put列的元素大小为0.1的矩阵
    Wx_plus_b=tf.matmul(inputs,Weight)+biases#做矩阵的乘法
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
#print('a',i)
a=[]
#y_data=np.square(x_data)-0.5+noise#在原来的计算基础上加上噪点，使他更和实际相匹配。
xs=tf.placeholder(tf.float32,[None,1])#表示输出为1，输入随机,一定要定义placeholder的类型，不然会报错。
ys=tf.placeholder(tf.float32,[None,1])
##定义隐藏层
l1=add_layer(xs,1,100,activation_function=tf.nn.relu)#隐藏层的输入为xs的size=1,输出定义为10个神经元(xs在最低下已经赋给x_data,x_data的size=1)
##定义输出层
prediction=add_layer(l1,100,1,activation_function=None)#输出层的输入为l1的size=10，输出定义为1个神经元
for i in range(b.shape[0]):
    x_data=b[1,1:701]
    #print('x_data',x_data)
    x_data = np.array(x_data)[:, np.newaxis]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,"D:/tensflow/my_net2/save_net.ckpt")
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        a.append(prediction_value)
result=np.array(a).reshape((466,700))
#print(result.shape)
data1 = pd.DataFrame(result)
data1.to_csv('data2.csv')