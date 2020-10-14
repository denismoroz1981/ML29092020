# import libraries
import numpy as np, pandas as pd
import scipy
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as statsimport
import pymc3 as pm3
#import numdifftools as ndt
import statsmodels.api as sm
#from statsmodels.base.model import GenericLikelihoodModel
import matplotlib.pyplot as plt


# define likelihood function
def MLERegression(params):
    intercept, beta, sd = params[0], params[1], params[2] # inputs are guesses at our parameters
    yhat = intercept + beta*x # predictions# next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of sd
    negLL = -np.sum(scipy.stats.norm.logpdf(y, loc=yhat, scale=sd) )# return negative LL
    return(negLL)


# generate data
N = 20
x = np.linspace(0,20,N)
eps = np.random.normal(loc = 0.0, scale = 1.0, size = N)
y = 3*x + 1+eps
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 0

#sns.regplot(df.x, df.y)
#plt.scatter(df.x, df.y)
#plt.show()

# split features and target
X = df[['constant', 'x']] # fit model and summarize
print(sm.OLS(y,X).fit().summary())


# let’s start with some random coefficient guesses and optimize
guess = np.array([5,5,2])
results = minimize(MLERegression, guess, method='Nelder-Mead', options={'disp': True})

print(results)





