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

* 비지도 학습은 {X} 자체가 가지고 있을 패턴을 찾아내는 데에 목적을 둔 방법이다.

또 지도학습은, 라벨 y의 형태에 따라, regression과 classification으로 나눌 수 있으며, 다음과 같이 크게 나눈다.
1. regression (some number)
2. Binary Classification (1 or 0)
3. Multi-class Classification (A, B, C, D, E, etc)

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



#### iii. Linear Regression by Tensorflow

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

Logistic Regressoin은 쉽게 설명해 Linear Regression model에서 나온 결과를 Sigmoid라는 함수를 통과시켜 얻은 값을 통해, 분류를 하는 모델이다.
$$
\begin{align}
Logistic Regression(x) &= \sigma(Linear Regression(x))\\
&= \sigma(XW+b)
\end{align}
$$
#### i. Sigmoid 함수

$$
\sigma(x) = \dfrac{1}{1-e^{-x}}
$$
x값이 0일 때는 0.5이며, 무한히 커질때는 1이 되고, 무한이 작아질때는, 0이 되는 **미분가능**한 함수이다.

이 형태 외에도, tanh(x) 등의 함수도 사용될 수 있다.



0에서 1사이의 값을 가지기 때문에, 확률로 취급할 수 있다.



Binary Classification의 두 라벨을 하나는 1 하나는 0이라고 할 때, 위의 Logistic Regression 모델을 이용해 분류할 수 있다.



#### ii. CostFunction for Logistic Regression

위 모델을 학습시키기 위해서는, 적절한 형태의 비용함수 (Cost Function)가 필요하다. 
$$
Cost(x) = -\dfrac{1}{m}\Sigma_i^m[y_ilog(RL(x_i))+(1-y_i)log(1-RL(x_i))]
$$


**Likelihood**

데이터가 주어질 때, 해당 클래스로 결정될 확률.
$$
\begin{align}
P(Y|X;\theta) &= \prod_i^NP(y_i|x_i;\theta)
\end{align}
\\
\text{i: data point}
$$




**In the case of binary-class under IID condition(Independent and identically distributed)** 
$$
\begin{align}
P(Y|X;\theta) &= \prod_i^NP(y_i|x_i;\theta)\\

&= \prod_i^N 
\begin{cases} 
P(y_i|x_i)\text{ when } y_i=1 \\ 
1-P(y_i|x_i)\text{ when } y_i=0
\end{cases}  \\
&=\prod_i^N P(y_i|x_i)^{y_i}(1-P(y_i|x_i))^{1-y_i}
\end{align}
$$


**Log-Likelihood**
$$
\begin{align}
logP(Y|X;\theta)&=log\prod_i^N P(y_i|x_i)^{y_i}(1-P(y_i|x_i))^{1-y_i}\\
&= \sum_i^N \Big( y_ilogP(y_i|x_i)+(1-y_i)log(1-P(y_i|x_i)) \Big)
\end{align}
$$

**And,**
$$
\begin{align}
\text{max }logP(Y|X;\theta) = \text{min }-logP(Y|X;\theta)\\ \\
\text{Since }P(y_i|x_i) = \sigma(W*X) \text{ in Logistic Regression model,}\\
\text{The cost function above is spontaneously derived on a way of maximizing Log-Likelihood}
\end{align}
$$


#### iii. Classification by Logistic Regression

위에서 언급했듯이 Sigmoid 함수를 통과한 값은 0과 1 사이에 들어가게 됨으로, 확률로써 취급될 수 있고, 0.5보다 크면 1, 더 작으면 0인 클래스로 분류할 수 있다.



사실 이는 Sigmoid 함수를 통과하기 전 선형변환 값이 0보다 큰 경우와 같다.
$$
\begin{align}
\dfrac{1}{1+e^{-x}} &> 0.5\\
1+e^{-x} &> 2\\
e^{-x} &> 1\\
x &> 0\\
\end{align}
$$


#### iv. Remarks in Implementation

log함수는 0에 가까울 때, 발산하게 된다.

그리고 지수함수는  급격히 증가하는 함수이다.

이 두가지 요소가 계산에 있어서 overflow를 발생하기 쉽게 한다. 그래서 위의 비용함수를 아래와 같이 정리할 수 있다.


$$
\begin{align}
Cost(x) &= -\dfrac{1}{m}\Sigma_i^m[y_ilog(RL(x_i))+(1-y_i)log(1-RL(x_i))]\\
-y_ilog(RL(x_i))+(1-y_i)log(1-RL(x_i)) 
&=
-y_ilog(\dfrac{1}{1+e^{-WX}})-(1-y_i)log(1-\dfrac{1}{1+e^{-WX}}) \\
\\
\text{let WX = m,}\\
&=-y_ilog(\dfrac{1}{1+e^{-m}})-(1-y_i)log(\dfrac{e^{-m}}{1+e^{-m}}) \\
&=y_ilog(1+e^{-m})+(1-y_i)(log(1+e^{-m})-log(e^{-m})) \\
&=m-y_i*m  +log(1+e^{-m}) \\
\\
\\
\text{applying } +log(e^{m})-log(e^{m}) \text{ on the RHS}
\\
m-y_i*m  +log(1+e^{-m}) &= m-y_i*m  +log(1+e^{-m})+log(e^{m})-log(e^{m})\\
&= -y_i*m +log(1+e^m)
\\
\end{align}
$$


**If we use above eqs when m>=0 and below one when m<0, it can be expressed to a single eqs.**




$$
max(m,0)-y_i*m+log(1+e^{-abs(m)})
$$
**This equation is safe from overflow!!!**



#### iv. Logistic Regression in TensorFlow

```python
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer

# load data
train_data = load_breast_cancer()
x_train = train_data.data
y_train = train_data.target
y_train = np.reshape(y_train, [-1,1])
print("shape of data features: {}".format(x_train.shape))
print("shape of data labels: {}".format(y_train.shape))


# Model
# tensorflow model using gradientTape with low API

X = tf.placeholder(tf.float32, shape=[None,30])
Y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random.normal([30,1],0,1))
b = tf.Variable(tf.zeros([1]))

linear_model = tf.matmul(X, w) + b

# Define Cost Function
##cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=linear_model))
cost = tf.reduce_mean(tf.maximum(linear_model,0)-linear_model*Y + tf.log(1+tf.exp(-tf.abs(linear_model))))


# HyperParameters
learning_rate = 0.0001
iter_num = 2000

optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
predict = tf.cast(linear_model > 0, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))


# training
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    cost_list = []
    acc_list = []
    for idx in range(iter_num):
        sess.run(optm,feed_dict={X:x_train, Y:y_train})
        
        if idx%100:
            pass
        else:
            tmp_cost, tmp_acc = sess.run([cost, acc], feed_dict={X:x_train, Y:y_train})
            cost_list.append(tmp_cost)
            acc_list.append(tmp_acc)
            print("# {}th iteration:".format(idx+1))
            print("cost : {}, accuracy: {:2f}%\n".format(tmp_cost, tmp_acc))
    print("iteraion has done!")
```

![](./imgs/logisticRegression.png)

breast-cancer data를 logistic regression을 활용해서 학습시켜보았다.

마지막에 정확도가 90%정도 나온 것을 볼 수 있다. 

- 정확도 metric으로만 판단할 수는 없다.
- Exploratory Data Analysis (EDA)를 통해 데이터의 분포와 특성을 파악한 뒤 적절한 metric을 통해 확인하여야 한다.



## IV. Tensorflow



[description regarding tensorflow]



#### I. Tensorflow Basics

Core Procedure in TensorFlow

1. **tf.Graph**
2. **tf.Session**

#### i. tf.Graph

This is computational Graph that consists of *operations*(Nodes) and *Tensors*(Edges).



##### TensorBoard

Utility to visualize the graph.



Use of graph

```python
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()
```



#### ii. tf.Session

When you request the output of a node with `Session.run` TensorFlow backtracks through the graph and runs all the nodes that provide input to the requested output node.



##### 1. Feeding

`tf.Graph` can be parameterized using `tf.placeholder`, and `feed_dict` argument in `sess.run()`



##### 2. Datasets

`tf.placeholder` works for simple experiments.

`tf.data` are preferred method to stream data into a model.



To get `tf.Tensor` from a Dataset.

1. convert data to `tf.data.Iterator`
2. then, call `tf.data.Iterator.get_next()`

example

```python
train_data = #...

slices = tf.data.Dataset.from_tensor_slices(train_data)
next_item = slices.make_one_shot_iterator().get_next() # get a row at train_data
# .make_ont_shot_iterator() returns tf.data.Iterator
with tf.Session() as sess:
    while True:
        try:
            print(sess.run(next_item))
        except: tf.errors.OutOfRangeError
            break

```

If it reaches end of the data, it throws `tf.errors.OutOfRangeError`



If Dataset depends on stateful operation like `tf.random.normal()`, it must be initialized as follows.

```python
r = tf.random.normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break
```



See Importing Data

[High level API importing data](<https://www.tensorflow.org/guide/datasets>)

##### 3. Layers (`tf.layers`)

Layer is predefined combination with sorts of operation and variables



example

```python
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```

**Remarks**

1. Function, whose name starts with capital (class), returns callable `?!` 

2. Variables in layer must be initialized as follows

```python
init = tf.global_variables_initializer()
sess.run(init)
```



##### 4. Layer Function shortcuts

For each layer class, tf also provides a shortcut function that can be run in a single call.

```python
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
```

While convenient, this approach allows no access to the [`tf.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/layers/Layer) object. This makes introspection and debugging more difficult, and layer reuse impossible.



##### 5. Loss (`tf.losses`)

`tf.losses` module provides several common loss functions.

```python
loss1 = tf.losses.mean_squared_error(labels= , predictions= )
loss2 = #...
```



##### 6. Training

1. make Optimizer (`tf.train.Optimizer`)  `tf.train.GradientDescentOptimizer`
2. `.minimize({loss})`

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```








## V. Basic Deep learing













## VI. TensorBoard









## VII. Keras in Tensorflow









## VIII. CNN Basics









## IX. CNN model for MNIST dataset



#### i. by using tensorflow





#### ii. by using keras in tf







## X. Representative CNN-based models and its applications











## XI. RNN basics

#### i. concept



#### ii. Word sentiment Classification







