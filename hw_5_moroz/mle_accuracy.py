# import libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as statsimport


# define likelihood function
def MLERegression(params):
    a, b, c, sd = params[0], params[1], params[2],params[3] # inputs are guesses at our parameters

    yhat = a*np.power(x,2)+b*x+c # predictions# next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of sd
    negLL = -np.sum(scipy.stats.norm.logpdf(y, loc=yhat, scale=sd) )# return negative LL
    return(negLL)

if __name__ == '__main__':
    # generating test data
    N = 20
    x = np.linspace(-10,10,N)
    np.random.seed(20) #fixing seed
    noise = np.random.normal(loc = 0.0, scale = 50.0, size = N) #adding noise
    a,b,c = 3, 2, 1
    y = a*np.power(x,2)+b*x+c+noise #function to generate test data

    # setting layout
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))
    fig.suptitle(f"Quadratic finction {a}*x^2+{b}*x+{c}+noise",y=0.99) #adding title
    fig.tight_layout() #improving layout
    axs = [ax1, ax2, ax3] #plot areas to iterate over

    #selecting parameters
    guess = np.array([[-5,-5,-5,-5],[0,0,0,0],[5,5,3,1]]) #initial (a priori) parameters
    methods={"TNC":"r","Powell":"b","COBYLA":"g"} #methods to test


    for i in range(3):
       axs[i].set_title(f"initials: {guess[i]}",fontsize=10) #subplot title with initial values
       axs[i].scatter(x, y, label="test range") #ploting real data
       for m, c in methods.items():
            #finding parameters of approximating function
            results = minimize(MLERegression, guess[i], method=m, options={'disp': True})
            #finding estimated values
            yhat1 = results.x[0]*np.power(x,2)+results.x[1]*x+results.x[2]
            #calculating MSE accuracy metric
            mse = np.sum(np.power(y-yhat1,2))/N
            #plotting data
            axs[i].tick_params(axis='both', which='major', labelsize=7)
            axs[i].plot(x,yhat1,color=c,label=f"{m}, mse={mse:.0f}")
            axs[i].legend(loc="upper center",fontsize=9) #adding legend

    plt.savefig("figures/methods_test.png")
    #plt.show()

