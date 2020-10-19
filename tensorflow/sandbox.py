import scipy.stats
import numpy as np

y=np.linspace(1,3,20)
yhat = 2

negLL = -scipy.stats.norm.logpdf(y, loc=yhat, scale=1)

print(negLL)


