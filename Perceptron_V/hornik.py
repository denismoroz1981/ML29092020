import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf



X = np.linspace(0,1,100)
w1 =1000
s1 = 0.4
b1 = -w1*s1

w2 =1000
s2 = 0.6
b2 = -w2*s2

w21=0.8
w22=-0.8

y = tf.math.sigmoid(w1*X+b1)*w21+tf.math.sigmoid(w2*X+b2)*w22

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
ax1.plot(X,y)
plt.show()