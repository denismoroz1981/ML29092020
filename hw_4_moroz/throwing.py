#importing packages
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

#function for mean and confidence interval culculation
def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

#function for distance calculation
def dist (v=30,a=60,v_distrib="norm", a_distrib="norm"):
    # downward acceleration
    g = 9.8
    # for the case without noise
    if v_distrib == "stable": x = v
    if a_distrib == "stable": y = a
    # drawing distributions with noise fixing the parameters of random numbers generator
    np.random.seed(40)
    if v_distrib == "norm": x = np.random.randn(1001)+v
    if a_distrib == "norm": y = np.random.randn(1001)+a
    if v_distrib == "unif": x = np.random.uniform(-10,10,1001)+v
    if a_distrib == "unif": y = np.random.uniform(-10,10,1001)+a
    # distance v^2*sin(2*a)/g, transformating degrees to radians
    s=np.power(x,2)*np.sin(2*y*np.pi/180)/g
    return s

#starting the program
if __name__ == '__main__':

    #setting params
    x = np.linspace(0, 1000, 1001)  #generating vector of attempts
    v = 30 #muzzle velocity in m/sec
    a = 60 #angle in degree

    #distances with combinations of distrubution types for velocity and angle

    s_s_s = dist(v,a,"stable","stable") #stable velocity and angle

    #drawing histogram
    plt.hist(s_s_s,bins=100)
    plt.title("Fig.1. Histogram of throws: v = 30 v/s, a = 60 degrees",fontweight="bold")
    plt.xlabel('Attempts')
    plt.ylabel('Frequency')

    # saving figure
    plt.savefig('figures/f_stable.png')
    plt.show()

    s_n_n = dist(v,a,"norm","norm")
    s_n_u = dist(v,a,"norm","unif")
    s_u_n = dist(v,a,"unif","norm")
    s_u_u = dist(v,a,"unif","unif")

    #setting layout
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(5,5))
    fig.suptitle("Combination of normal and uniform distributions of velocity (v) and angle (a)",
                 fontsize=9)
    fig.tight_layout() #improving spacing

    # drawing distances obtained

    #calculating meand and confidence interval for the second graph
    m, ci1, ci2 = mean_confidence_interval(s_n_n)[0],mean_confidence_interval(s_n_n)[1],mean_confidence_interval(s_n_n)[2]
    sd = np.std(s_n_n) #standard deviation
    ax1.hist(s_n_n,bins=100) #drawing histogram
    ax1.set_title("Fig.2. v ~ N(30,1), a ~ N(60,1)",fontsize=8,fontweight="bold") #adding title
    ax1.tick_params(axis='both', which='major', labelsize=7) #smaller ticks labels
    ax1.axvline(m,color='k', linestyle='dashed', linewidth=1) #adding mean line
    ax1.text(m * 1.05, ax1.get_ylim()[1]*0.9, 'Mean: {:.2f}'.format(m),
             fontsize=7) #adding text lable with mean value
    ax1.text(m * 1.05, ax1.get_ylim()[1] * 0.8, 'SD: {:.2f}'.format(sd),
             fontsize=7)  # adding text lable with standard deviation
    ax1.text(m * 1.05, ax1.get_ylim()[1] * 0.7, 'Confidence: 99%',fontsize=7)  # adding text lable with confidence
    ax1.axvline(ci1, color='r', linestyle='dashed', linewidth=0.5)  # adding confidence interval
    ax1.axvline(ci2, color='r', linestyle='dashed', linewidth=0.5)  # adding confidence interval

    #the third graph
    m = s_n_u.mean()
    sd = np.std(s_n_u)
    ax2.hist(s_n_u,bins=100)
    ax2.set_title("Fig.3. v ~ N(30,1), a ~ U[50,70]",fontsize=8,fontweight="bold")
    #ax2.set_yticklabels([]) #removing y-axis ticks
    ax2.tick_params(axis='both', which='major', labelsize=7)  # smaller ticks labels
    ax2.axvline(m, color='k', linestyle='dashed', linewidth=1)  # adding mean line
    ax2.text(m * 1.05, ax2.get_ylim()[1] * 0.9, 'Mean: {:.2f}'.format(m),
             fontsize=7)  # adding text lable with mean value
    ax2.text(m * 1.05, ax2.get_ylim()[1] * 0.8, 'SD: {:.2f}'.format(sd),
             fontsize=7)  # adding text lable with standard deviation

    # the forth graph
    m = s_u_n.mean()
    sd = np.std(s_u_n)
    ax3.hist(s_u_n,bins=100)
    ax3.set_title("Fig.4. v ~ U[20,40], a ~ N(60,1)",fontsize=8,fontweight="bold")
    ax3.tick_params(axis='both', which='major', labelsize=7)  # smaller ticks labels
    ax3.axvline(m, color='k', linestyle='dashed', linewidth=1)  # adding mean line
    ax3.text(m * 1.05, ax3.get_ylim()[1] * 0.9, 'Mean: {:.2f}'.format(m),
             fontsize=7)  # adding text lable with mean value
    ax3.text(m * 1.05, ax3.get_ylim()[1] * 0.8, 'SD: {:.2f}'.format(sd),
             fontsize=7)  # adding text lable with standard deviation

    # the fifth graph
    m = s_u_u.mean()
    sd = np.std(s_u_u)
    ax4.hist(s_u_u,bins=100)
    ax4.set_title("Fig.5. v ~ U[20,40], a ~ U[50,70]",fontsize=8,fontweight="bold")
    #ax4.set_yticklabels([])  # removing y-axis ticks
    ax4.tick_params(axis='both', which='major', labelsize=7)  # smaller ticks labels
    ax4.axvline(m, color='k', linestyle='dashed', linewidth=1)  # adding mean line
    ax4.text(m * 1.05, ax4.get_ylim()[1] * 0.9, 'Mean: {:.2f}'.format(m),
             fontsize=7)  # adding text lable with mean value
    ax4.text(m * 1.05, ax4.get_ylim()[1] * 0.8, 'SD: {:.2f}'.format(sd),
             fontsize=7)  # adding text lable with standard deviation

    #saving figures
    plt.savefig('figures/f_comb_distrib.png')
    #showing figures
    plt.show()


