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
#tạo dữ liệu giả lập và thêm nhiễu vào nhìn cho thực tế
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
#sử dụng Normal Equation  đúng theo công thức
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
#đường màu xanh lá là đường gốc dùng để giả lập dữ liệu
#đường màu đỏ là mới tính bằng công thức ra, 2 đường rất sát với nhau


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
#2 kq gần như là giống nhau sai số rất ít chắc là do dữ liệu còn nhỏ và đơn giản