import matplotlib.pyplot as plt
import numpy as np


#preparing layout
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(16,2))
#plt.axes()

x=np.linspace(-5,5,1000)

#Logistics function
def f_logist(x): return 1/(1+np.exp(-x))
f_logist2=np.vectorize(f_logist)

ax1.plot(x,f_logist2(x))
ax1.set_title("Logistic function",fontsize=10)
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax1.spines['left'].set_position('center')
ax1.spines['bottom'].set_position('zero')
# Eliminate upper and right axes
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
# Show ticks in the left and lower axes only
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
#changing ticks density
ax1.yaxis.set_major_locator(plt.MaxNLocator(3))

#SoftPlus function
def f_softplus(x): return np.log(1+np.exp(x))
f_softplus2=np.vectorize(f_softplus)

ax2.plot(x,f_softplus2(x))
ax2.set_title("SoftPlus function",fontsize=10)
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax2.spines['left'].set_position('center')
ax2.spines['bottom'].set_position('zero')
# Eliminate upper and right axes
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
# Show ticks in the left and lower axes only
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

#ArcTan function
def f_arctan(x): return np.arctan(x)
f_arctan2=np.vectorize(f_arctan)

ax3.plot(x,f_arctan2(x))
ax3.set_title("ArcTan function",fontsize=10)
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax3.spines['left'].set_position('center')
ax3.spines['bottom'].set_position('zero')
# Eliminate upper and right axes
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
# Show ticks in the left and lower axes only
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')

plt.savefig('figures/f_act_fig.png')
plt.show()


