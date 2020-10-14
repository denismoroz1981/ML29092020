import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np

#y = np.random.random(10000)
y=np.random.randn(10000)
plt.hist(y,bins=10)

plt.show()
