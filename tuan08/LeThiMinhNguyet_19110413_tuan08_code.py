'''
The following code is mainly from Chap 4, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb

LAST REVIEW: April 2021
'''

# In[0]: IMPORTS, SETTINGS
#region
import sys
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import numpy as np
import os   
np.random.seed(42) # to output the same result across runs
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)       
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd") # Ignore useless warnings (see SciPy issue #5998)
font_size = 14
let_plot = True
#endregion


''' WEEK 07 '''

# In[1]: LINEAR REGRESSION USING NORMAL EQUATION 
# 1.1. Generate linear-looking data 
n_samples = 150
X = 2*np.random.rand(n_samples, 1) # random real numbers in [0,2]
y_no_noise = 3 + 7*X; 
y = y_no_noise + np.random.randn(n_samples, 1) # noise: random real numbers with Gaussian distribution of mean 0, variance 1

let_plot = True
if let_plot:
    plt.plot(X, y, "k.")
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.ylabel("$y$", rotation=0, fontsize=font_size)
    plt.axis([0, 2, 0, 15])
    #plt.savefig("figures/generated_data_plot",format='png', dpi=300)
    plt.show()


#%% 1.2. Compute Theta using Normal Equation 
X_add_x0 = np.c_[np.ones((n_samples, 1)), X]  # add x0 = 1 to each instance
theta_norm_eq = np.linalg.inv(X_add_x0.T @ X_add_x0) @ X_add_x0.T @ y 
# Note: Theta is a bit different from the true parameters due to the noise.
	
# 1.3. Try prediction 
X_test = np.array([[0], [2], [15]]) # 3 instances
X_test_add_x0 = np.c_[np.ones((len(X_test), 1)), X_test]  # add x0 = 1 to each instance
y_predict = X_test_add_x0 @ theta_norm_eq

# 1.4. Plot hypothesis 
if let_plot:
	plt.plot(X, y_no_noise, "g-",label="True  hypothesis")
	plt.plot(X_test, y_predict, "r-",label='Hypothesis')
	plt.plot(X, y, "k.",label='Training sample')
	plt.axis([0, 2, 0, 15])
	plt.legend()    
	plt.xlabel("$x_1$", fontsize=font_size)
	plt.ylabel("$y$", rotation=0, fontsize=font_size)
	plt.show()	


# In[2]: LINEAR REGRESSION USING GRADIENT DESCENT 

# 2.1. Gradient descent (>> see slide)

#%% 2.2. Batch gradient descent
eta = 0.1  # learning rate
m = len(X)
np.random.seed(42);
theta_random_init = np.random.randn(2,1)
theta = theta_random_init  # random initialization
#for iteration in range(1,1000) # use this if you want to stop after some no. of iterations, eg. 1000
while True:
	#gradients = 2/m * X_add_x0.T @ (X_add_x0 @ theta - y); # WARNING: @ (mat multiply) causes weird indent errors when running in Debug interactive
	gradients = 2/m * X_add_x0.T .dot (X_add_x0 .dot (theta) - y); # works the same at the code above, but no indent errors
	theta = theta - eta*gradients;
	if (np.abs(np.mean(eta*gradients)) < 0.000000001): 
		break # stop when the change of theta is small

# 2.3. Compare with theta by Normal Eq.
theta_norm_eq
theta_BGD = theta


''' WEEK 08 '''

#%% 2.4. Learning rates (>> see slide)

# 2.5. Try different learning rates 
#hàm vẽ ra cái gradient descent vẽ ra sự thay đổi
def plot_gradient_descent(theta, eta, theta_path=None, n_iter_plot=10, n_iter_run=1000):
    m = len(X_add_x0)
    plt.plot(X, y, "k.")
    for iteration in range(1,n_iter_run): # run 1000 iter, instead of convergence stop
        if iteration <= n_iter_plot:
            y_predict = X_test_add_x0 .dot (theta) 
            if iteration == 1:
                plt.plot(X_test, y_predict, "g--", label="initial theta",linewidth=2)
            elif iteration == n_iter_plot: 
                plt.plot(X_test, y_predict, "r-", label="theta at 10th iter",linewidth=2)  
            else:
                plt.plot(X_test, y_predict, "b-",linewidth=2)                  
        gradients = 2/m * X_add_x0.T .dot (X_add_x0 .dot (theta) - y)
        theta = theta - eta*gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.axis([0, 2, 0, 15])
    plt.legend(loc='upper right');
    plt.title(r"$\eta = {}$".format(eta), fontsize=font_size)

#vẽ các BGD nhỏ vừa lớn
# 2.5.1. Plot BGD  with small learning rate
np.random.seed(42)
init_theta = np.random.randn(2,1)  # random initialization
fig = plt.figure(figsize=(10,5))
plt.subplot(131);
plot_gradient_descent(init_theta, eta=0.02); plt.ylabel("$y$", fontsize=font_size)

# 2.5.2. Plot BGD with good learning rate 
plt.subplot(132); 
theta_path_bgd = [theta_random_init]
plot_gradient_descent(init_theta, eta=0.1, theta_path=theta_path_bgd)

# 2.5.3. Plot BGD with large learning rate
plt.subplot(133); 
plot_gradient_descent(init_theta, eta=0.5)
fig.suptitle("Theta values in the first 10 iterations of BGD", fontsize=14)
plt.show()



#%% 2.6. How to find a good learning rate eta?
#   1. Try small learning rate, then increase it gradually.
#   2. Do hyperparameter TUNING for eta (using, e.g., grid search).


''' ________________________________________________ '''


# In[3]: STOCHASTIC GRADIENT DESCENT 

# 3.0. Problems of gradient descent (>> see slide)

# 3.1. Stochastic gradient descent (>> see slide)

# 3.2. Training using Stochastic GD
#hàm giúp giảm dần theo thời gian
def learning_schedule(t):
    alpha = 0.2; t0 = 50; # learning schedule hyperparameters
    eta = 1 / (alpha* (t + t0))
    return eta

m = len(X_add_x0)
theta = theta_random_init  # random initialization
theta_path_sgd = [theta_random_init]  

n_epochs = 50 # << 1 epoch = 1 time of running m iter (m: no. of training samples)
for epoch in range(n_epochs):
    for i in range(m):
        # Just for plotting purpose
        if epoch == 0 and i <= 20:                       
            y_predict = X_test_add_x0.dot(theta)   
            if i == 0:
                plt.plot(X_test, y_predict, "g--", label="initial theta",linewidth=2)
            elif i == 20: 
                plt.plot(X_test, y_predict, "r-", label="theta at 20th iter",linewidth=2)  
            else:
                plt.plot(X_test, y_predict, "b-",linewidth=2)     
        # Pick a random sample
        random_index = np.random.randint(m)
        xi = np.array([X_add_x0[random_index]])
        yi = np.array([y[random_index]])
        # Compute gradients
        gradients = 2 * xi.T .dot (xi .dot (theta) - yi)
        # Compute learning rate
        eta = learning_schedule(m*epoch + i)
        # Update theta
        theta = theta - eta*gradients
        theta_path_sgd.append(theta)        
plt.plot(X, y, "k.")                                 
plt.xlabel("$x_1$", fontsize=font_size)                   
plt.ylabel("$y$", fontsize=font_size)            
plt.axis([0, 2, 0, 15])  
plt.legend()
plt.title('SDG in the first 20 iter of the first epoch')
plt.show()                                            

# 3.3. Compare thetas found by SGD and BGD 
theta_BGD
theta_SGD = theta


# In[4]: MINI-BATCH GRADIENT DESCENT

# 4.1. (>> see slide)

# 4.2. Implement and run mini-batch GD
def learning_schedule(t):
    t0, t1 = 200, 1000
    return t0 / (t + t1)

n_epochs = 50
minibatch_size = 20
theta = theta_random_init  # random initialization
theta_path_mgd = [theta_random_init]
t = 0      
#mỗi bước chạy cứ tính theo công thức thôi     
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_addx0_shuffled = X_add_x0[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        # Get random samples
        xi = X_addx0_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T .dot (xi .dot (theta) - yi)
        # Compute learning rate
        t += 1
        eta = learning_schedule(t)
        # Update theta
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

# 4.3. Plot update paths of BGD, SGD, and Mini-batch GB
#vẽ ra để so sánh cả 3 phương pháp
if let_plot:
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)

    plt.figure(figsize=(7,4))
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "g-o", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "b-s", linewidth=1, label="Mini-batch")
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "r-|",markersize=7, linewidth=2, label="Batch")
    plt.legend(loc="upper left", fontsize=font_size)
    plt.xlabel(r"$\theta_0$", fontsize=font_size)
    plt.ylabel(r"$\theta_1$   ", fontsize=font_size, rotation=0)
    #plt.axis([2.5, 4.5, 2.3, 3.9])   
    plt.show()

# 4.4. Comparison of training algorithms (>> see slide) 
print("\n")


# In[5]: IMPLEMENTATION BY SCIKIT-LEARN 
#cài đặt bằng scikit learn
# 5.1. Sklearn implementation of Normal Equation (using SVD)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# Training
lin_reg.fit(X, y.ravel()) # ravel(): convert to 1D array                  
# Learned parameters (theta)
lin_reg.intercept_, lin_reg.coef_   
# Compare with theta by previous implementation
theta_norm_eq 
# Prediction
lin_reg.predict(X_test)            

#%% 5.2. Sklearn implementation of Stochastic Gradient Descent 
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-9, learning_rate='optimal', alpha=0.2, random_state=42, penalty=None)
# Training
sgd_reg.fit(X, y.ravel())
# Learned parameters (theta)
sgd_reg.intercept_, sgd_reg.coef_
# Compare with theta by previous implementation
theta_SGD # different result due to no control on t0 in the learning schedule.
# Prediction
sgd_reg.predict(X_test)   