# DeepLearning for everyone by SungKim, ver2
[해당 강의 Docker Guide](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/docker_user_guide.md)
[Tensorflow codes](https://github.com/deeplearningzerotoall/TensorFlow)

Docker를 이용한 환경설정

* window
Docker for windows, or docker toolbox
If `docker run hello-world` returns `hello from docker` at docker-quickstart-terminal, it is installed successfully.


* MacOS
Just Install Docker on the website

* Linux
```
curl -fsSl https://get.docker.com > docker.sh
sudo sh docker.sh
```

## I. ML Basics
머신러닝은 큰 분류로 Supervised Learning(지도 학습)과 Unsupervised Learning(비지도 학습)으로 나눌 수 있고,

* 지도 학습은 라벨이 존재하는 데이터 셋 {(X,y)}에 대해서 X 데이터를 y 라벨로 매핑시키는 모델을 학습하는 것을 의미하며,

* 비지도 학습은 {X} 자체에 있는 패턴을 찾아내는 데에 목적을 둔 방법이다.

또 지도학습은, 라벨 y의 형태에 따라, regression과 classification으로 나눌 수 있으며, 다음과 같이 크게 나눈다.
1. regression (some number)
2. Binary Classification (1 or 0)
3. Multivariable Classification (A, B, C, D, E, etc)

## II. Linear Regression
선형회귀 모델은 가장 단순하지만, 이 모델을 통해 ML을 관통하는 핵심적인 개념을 습득할 수 있다.

선형 회귀 모델
$$
H(x) = W*x+b
$$

b를 W와 x에 넣어 간단히 $ H(x) = W*x$로 표현할 수 있다.

주어진 가정 $H(x)$가 y와 최대한 일치하도록 하는 것이 위 모델의 목적이다.

이를 위해 아래와 같은 과정들이 거쳐지게 된다.



1. 모델 구성
2. 비용함수 구성
3. Gradient Descent Method를 통해, W를 구함.



#### i. Cost Function

보통 선형회귀 모델로 Mean Squared Error가 사용된다.
$$
\begin{align*}
MSE(x) &= \dfrac{1}{m} \Sigma (y-H(x))^2 \\
Cost(x) &= \dfrac{1}{m} \Sigma_i^m (y_i-W*x_i)^2 \\
\end{align*}
$$
where m = number of data 



위의 비용함수를 최소화 시키는 W를 찾는 것이 목적이다.

#### ii. Gradient Descent Method

데이터의 수가 적을 때는, Moore-Penrose Inverse (Pseudo Inverse)를 활용해  MSE를 최소화 하는 W를 정확히 구할 수 있지만, 데이터 수가 많아질 수록 계산량이 급격히 커지게 된다. 그러므로 근사적인 해를 구하는 여러 방법들을 사용하게 되는데, 기본이 되는 대표적인 방법이 바로 Gradient Descent Method이다.



Gradient Descent Method
$$
x_{n+1} = x_{n} - \alpha \dfrac{\partial f}{\partial x}
$$
기울기의 부호 반대로, $ \alpha $ (learning rate)을 곱한 뒤 x를 업데이트 해주는 방법이다.

용어 그대로 기울기를 따라 하강하는 방법이다. (경사하강법)

![](./imgs/gradientDescent1.png)



이때, 학습률 $\alpha$ 를 사람이 임의로 주게 되는데 이를 Hyper Parmeter라 한다. 이 방법 외에도, Momentum, Adam 등 여러 방법이 있지만, 모두 경사하강법에 기반한 식의 추가 및 변형이라 할 수 있다.



만약 학습률이 너무 클 경우에는, 발산할 수도 있다.

![](./imgs/gradientDescent2lglr.png)



또, 만약 학습률이 너무 작을 경우, 수렴이 너무 늦어 굉장히 비효율적일 수 있다.

![](./imgs/gradientDescent3smlr.png)

또, 최적의 x가 1개 이상인 경우에는 local minima에 빠지기 쉽다.



Linear Regression by Tensorflow

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 임시 데이터 셋
x_train = np.reshape(np.array([4,7,8,9,1,6]), [-1,1])
y_train = np.reshape(np.array([8,14,16,18,2,12]), [-1, 1])

# Hyper Parameters
alpha = 0.01
iter_num = 100

# tf placeholder
X = tf.placeholder(tf.float32, shape=[None,1])
Y = tf.placeholder(tf.float32, shape=[None,1])

# tf Variable (our object)
W = tf.Variable(tf.random.normal([1],0,1))
b = tf.Variable(0.0)

# Hypothesis
model = tf.matmul(X,W)+b  # X then, W

# Cost
cost = tf.reduce_mean(tf.pow(Y-model, 2))

# Optimizer(Gradient Descent)
optm = tf.train.GradientDescentOptimizer(learning_rate = alpha)

train = optm.minimize(cost)

with tf.Session() as sess:
    # Initializaing variables
    sess.run(tf.global_variables_initializer())
    for idx in range(iter_num):
        sess.run(train,feed_dict={X:x_train, Y:y_train})
        
        if (idx) % 10:
            pass
        else:
            _cost = sess.run(cost, feed_dict={X:x_train, Y:y_train})
            print("{}th \n cost: {}".format(idx+1, _cost))
            
    print(sess.run(W), sess.run(b))
    
plt.scatter(x_train,y_train)
a = np.linspace(0,9,20)
plt.plot(a,a*1.96389+0.25381,'r')
plt.show()
```



![](./imgs/LR_result1.png)

보다 시피, 파란색 포인트들을 잘 표현하는 W와  b를 얻은 것을 알 수 있다. 

아래는 경사하강법이 진행됨에 따라 나타낸 Cost값이다.

값이 점점 작아지는 것을 확인할 수 있다.

![](./imgs/LR_cost1.png)

마지막에 얻은 W = 1.987, b = 0.093 은 의도한 정답인 (W=2, b=0) 과는 차이가 있지만, 근사적으로 정답에 접근하고 있는 것을 알 수 있다.



기계학습의 지도학습에서 학습을 시킨다는 것은 지금까지 거쳐온 과정을 진행하는 것과 개념적으로 크게 다르지 않다.

1. 데이터에서 x -> y를 위한 적절한 모델 설정
2. 비용함수(목적함수) 정의
3. Gradient Descent에 기반한 여러 방법 중 하나를 선택, HyperParameter를 잘 조절하여 모델의 Weight를 구함. 
4. 얻어진 Weight와 그 모델로 앞으로의 데이터 예측 등에 활용.



## III. Logistic Regression







## IV. Tensorflow

#### i . Basics







#### ii. Practices







## V. TensorBoard







## VI. Keras in Tensorflow







## VII. CNN Basics





## VIII. CNN model for MNIST dataset



#### i. by using tensorflow





#### ii. by using keras in tf



## IX. Representative CNN-based models and its applications





## X. RNN basics

#### i. concept



#### ii. Word sentiment Classification







