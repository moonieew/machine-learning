'''
The following code is mainly from Chap 5, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/05_support_vector_machines.ipynb

LAST REVIEW: Dec 2020
'''
#Cau 1:Thay đổi các phương trình tạo data giả lập, chọn các classes khác 
# trong Iris dataset, và thực hiện chạy từng bước các thuật toán 
# như demo trong bài học (làm tiếp theo tuần trước).   
# In[0]: IMPORTS, SETTINGS
import sklearn 
assert sklearn.__version__ >= "0.20" # sklearn ≥0.2 is required
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
np.random.seed(42) # to output the same across runs
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[1]: LINEAR SVM      
# 1.1. Load Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
versicolor_or_virginica = (y == 1) | (y == 2)
X = X[versicolor_or_virginica]  # use only 2 classes: virginica, versicolor
y = y[versicolor_or_virginica]

# 1.2. Decision boundaries of arbitrary models
x0 = np.linspace(0, 20, 200)
x1_model_1 = 5*x0 - 20
x1_model_2 = -x0 + 6

# 1.3. Train a SVM classifier model
#from sklearn.svm import SVC 
#svm_clf = SVC(kernel="linear", C=np.inf) 
from sklearn.svm import LinearSVC # faster than SVC on large datasets
svm_clf = LinearSVC(C=np.inf) # C: larger => less regularization. loss = 'hinge': standard loss
svm_clf.fit(X, y)
svm_clf.predict(X) # Predicted labels
svm_clf.decision_function(X) # Scores of samples

# 1.4. Plot decision boundaries of models
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # Plot decision boundary:
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]        
    x0 = np.linspace(xmin, xmax, 200)
    x1 = -w[0]/w[1]*x0 - b/w[1] # Note: At the decision boundary, w0*x0 + w1*x1 + b = 0 => x1 = -w0/w1 * x0 - b/w1
    plt.plot(x0, x1, "k-", linewidth=3, label="SVM")
    
    # Plot margins:
    margin = 1/w[1]
    gutter_up = x1 + margin
    gutter_down = x1 - margin
    plt.plot(x0, gutter_up, "k:", linewidth=2)
    plt.plot(x0, gutter_down, "k:", linewidth=2)
# Plot arbitrary model 1:
plt.figure(figsize = [16, 5])
plt.subplot(1,3,1);
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^", label="Iris versicolor")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go", label="Iris virginica")
plt.plot(x0, x1_model_1, "k-", linewidth=3)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.title("Decision boundary of model 1", fontsize=14)

# Plot arbitrary model 2:
plt.subplot(1,3,2);
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go")
plt.plot(x0, x1_model_2, "k-", linewidth=3, label="Model 2")
plt.xlabel("Petal length", fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.title("Decision boundary of model 2", fontsize=14)

# Plot SVM model:
plt.subplot(1,3,3);
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go")
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.xlabel("Petal length", fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.title("Decision boundary of SVM model", fontsize=14)
plt.savefig("figs/01_Decision boundaries.png")
plt.show()

# 1.5. Large margin classification (>> see slide)

# 1.6. Support vectors (>> see slide) 
let_plot = True
if let_plot:
    # Plot SVM model in a separate figure
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go")
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis("square")
    plt.axis([0, 5.5, -1, 2.5])
    plt.title("Decision boundary of SVM model", fontsize=14)
    plt.savefig("figs/02_Linear_SVM")
    plt.show()



# In[3]: NONLINEAR SVM 

# 3.1. Intro (>> see slide)

# 3.2. Load non-linear data
from sklearn.datasets import make_moons
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "rs")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "bo")
    plt.axis(axes)
    #plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)
if let_plot:
    plot_dataset(X, y, [2, 10, 0, 2.5])
    plt.savefig("figs/04_Nonlinear_data.png");
    plt.show()


# In[4]: METHOD 1 FOR NONLINEAR DATA: ADD POLINOMIAL FEATURES AND TRAIN LINEAR SVM
# 4.1. Add polinomial features and train linear svm 
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=40, loss="hinge", random_state=42)) ])
polynomial_svm_clf.fit(X, y)

# Plot decision boundary
def plot_predictions(clf, axes, no_of_points=500):
    x0 = np.linspace(axes[0], axes[1], no_of_points)
    x1 = np.linspace(axes[2], axes[3], no_of_points)
    x0, x1 = np.meshgrid(x0, x1)
    X = np.c_[x0.ravel(), x1.ravel()]

    # Plot predicted labels (decision boundary)
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.bwr, alpha=0.12)  
    
if True:
    plot_predictions(polynomial_svm_clf, [2, 10, 0, 2.5])
    plot_dataset(X, y, [2, 10, 0, 2.5])
    plt.show()

# 4.2. Kernel trick for method 1: Polynomial kernel
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
# NOTE: 
#   larger coef0 => the more the model is influenced by high-degree polynomials
poly_svm_1 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=0.5, C=1))  ]) 
poly_svm_1.fit(X, y)

poly_svm_2 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=100, C=5))  ])
poly_svm_2.fit(X, y)


if let_plot:
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plot_predictions(poly_svm_1, [2, 10, 0, 2.5])
    plot_dataset(X, y, [2, 10, 0, 2.5])
    plt.title(r"degree={}, coef0={}, C={}".format(poly_svm_1[1].degree,poly_svm_1[1].coef0,poly_svm_1[1].C), fontsize=14)

    plt.subplot(1,2,2)
    plot_predictions(poly_svm_2, [2, 10, 0, 2.5])
    plot_dataset(X, y, [2, 10, 0, 2.5])
    plt.title(r"degree={}, coef0={}, C={}".format(poly_svm_2[1].degree,poly_svm_2[1].coef0,poly_svm_2[1].C), fontsize=14)
    plt.ylabel("")
    plt.show()

#Cau 2: Huấn luyện các classifiers LinearSVC, SVC, SGDClassifier trên một dataset 
# tự tạo (dataset này có thể được phân lớp bằng linear boundary) và xem 
# 3 classifiers này có thể tạo ra cùng một model không. 
# Giải thích kết quả.
# In[1]: LINEAR SVM      
# 1.1. Load Iris dataset
from sklearn import datasets
wine = datasets.load_wine()
X = wine["data"][:, (0, 9)]  # alcohol, Colour Intensity
y = wine["target"]
class_1_or_class_2 = (y == 1) | (y == 2)
X = X[class_1_or_class_2]  # use only 2 classes: class_1; class_2
y = y[class_1_or_class_2]

# 1.3. Train a SVM classifier model
#from sklearn.svm import SVC 
#svm_clf = SVC(kernel="linear", C=np.inf) 
from sklearn.svm import LinearSVC # faster than SVC on large datasets
svm_clf = LinearSVC(C=np.inf) # C: larger => less regularization. loss = 'hinge': standard loss
svm_clf.fit(X, y)
svm_clf.predict(X) # Predicted labels
svm_clf.decision_function(X) # Scores of samples

# 1.4. Plot decision boundaries of models
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # Plot decision boundary:
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]        
    x0 = np.linspace(xmin, xmax, 200)
    x1 = -w[0]/w[1]*x0 - b/w[1] # Note: At the decision boundary, w0*x0 + w1*x1 + b = 0 => x1 = -w0/w1 * x0 - b/w1
    plt.plot(x0, x1, "k-", linewidth=3, label="SVM")
    
    # Plot margins:
    margin = 1/w[1]
    gutter_up = x1 + margin
    gutter_down = x1 - margin
    plt.plot(x0, gutter_up, "k:", linewidth=2)
    plt.plot(x0, gutter_down, "k:", linewidth=2)

# Plot arbitrary model 1:
x1_model_1=0.5*x0
plt.figure(figsize = [16, 5])
plt.subplot(1,3,1);
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^", label="Class 1")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go", label="Class 2")
plt.plot(x0, x1_model_1, "k-", linewidth=3)
plt.xlabel("Alcohol", fontsize=14)
plt.ylabel("Colour Intensity", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 20, 0, 10])
plt.title("Decision boundary of model 1", fontsize=14)

# Plot arbitrary model 2:
x1_model_2 = -0.1*x0 + 6
plt.subplot(1,3,2);
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go")
plt.plot(x0, x1_model_2, "k-", linewidth=3, label="Model 2")
plt.xlabel("Alcohol", fontsize=14)
plt.axis([0, 20, 0, 10])
plt.title("Decision boundary of model 2", fontsize=14)

# Plot SVM model:
plt.subplot(1,3,3);
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go")
plot_svc_decision_boundary(svm_clf, 0, 20)
plt.xlabel("Alcohol", fontsize=14)
plt.axis([0, 20, 0, 10])
plt.title("Decision boundary of SVM model", fontsize=14)
plt.savefig("figs/01_Decision boundaries.png")
plt.show()

# 1.5. Large margin classification (>> see slide)

# 1.6. Support vectors (>> see slide) 
let_plot = True
if let_plot:
    # Plot SVM model in a separate figure
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "go")
    plot_svc_decision_boundary(svm_clf, 0, 20)
    plt.xlabel("Alcohol", fontsize=14)
    plt.ylabel("Colour Intensity", fontsize=14)
    plt.axis("square")
    plt.axis([0, 20, 0, 10])
    plt.title("Decision boundary of SVM model", fontsize=14)
    plt.savefig("figs/02_Linear_SVM")
    plt.show()

# In[3]: NONLINEAR SVM 

# 3.1. Intro (>> see slide)

# 3.2. Load non-linear data
from sklearn.datasets import make_moons
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "rs")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "bo")
    plt.axis(axes)
    #plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)
if let_plot:
    plot_dataset(X, y, [0, 20, 0, 10])
    plt.savefig("figs/04_Nonlinear_data.png");
    plt.show()


# In[4]: METHOD 1 FOR NONLINEAR DATA: ADD POLINOMIAL FEATURES AND TRAIN LINEAR SVM
# 4.1. Add polinomial features and train linear svm 
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=40, loss="hinge", random_state=42)) ])
polynomial_svm_clf.fit(X, y)

# Plot decision boundary
def plot_predictions(clf, axes, no_of_points=500):
    x0 = np.linspace(axes[0], axes[1], no_of_points)
    x1 = np.linspace(axes[2], axes[3], no_of_points)
    x0, x1 = np.meshgrid(x0, x1)
    X = np.c_[x0.ravel(), x1.ravel()]

    # Plot predicted labels (decision boundary)
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.bwr, alpha=0.12)  
    

if let_plot:
    plot_predictions(polynomial_svm_clf, [0,20,0,10])
    plot_dataset(X, y, [0,20,0,10])
    plt.title("Linear SVC")
    plt.show()

# 4.2. Kernel trick for method 1: Polynomial kernel
from sklearn.svm import SVC
# NOTE: 
#   larger coef0 => the more the model is influenced by high-degree polynomials
poly_svm_1 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=2, coef0=1, C=5))  ]) 
poly_svm_1.fit(X, y)


poly_svm_3 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SGDClassifier(loss="hinge"))  ])
poly_svm_3.fit(X, y)

if let_plot:
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plot_predictions(poly_svm_1, [0,20,0,10])
    plot_dataset(X, y, [0,20,0,10])
    plt.title(r"SVC degree={}, coef0={}, C={}".format(poly_svm_1[1].degree,poly_svm_1[1].coef0,poly_svm_1[1].C), fontsize=14)

    plt.subplot(1,2,2)
    plot_predictions(poly_svm_3, [0,20,0,10])
    plot_dataset(X, y, [0,20,0,10])
    plt.title("SGD Classifer")
    plt.ylabel("")
    plt.show()


# In[5]: METHOD 2: ADD SIMILARITY FEATURES AND TRAIN 
# 5.1. Generate 1-fearture data (1-dimenstional data)
X_1D = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]).reshape(-1,1) 
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0]) # 2 classes

# 5.2. Plot Gaussian kernel graphs
def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)
def plot_kernel(X_1D,y,landmark,gamma, no_plot_points=200, xy_lim = [-4.5, 4.5, -0.1, 1.1]):  
    # Plot samples:
    plt.axhline(y=0, color='k') # Ox axis
    plt.plot(X_1D[y==0], np.zeros(4), "rs", markersize=9, label="Data samples (class 0)")
    plt.plot(X_1D[y==1], np.zeros(5), "g^", markersize=9, label="Data samples (class 1)")

    # Plot the landmark:
    plt.scatter(landmark, [0], s=200, alpha=0.5, c="orange")
    plt.annotate(r'landmark',xytext=(landmark, 0.2),
                 xy=(landmark, 0), ha="center", fontsize=14,
                 arrowprops=dict(facecolor='black', shrink=0.1)  )
    
    # Plot Gaussian kernel graph: 
    x1_plot = np.linspace(-4.5, 4.5, no_plot_points).reshape(-1,1)  
    x2_plot = gaussian_rbf(x1_plot, landmark, gamma)
    plt.plot(x1_plot, x2_plot, "b--", linewidth=2, label="Gaussian kernel")
    
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=13)
    #plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.axis(xy_lim)
    plt.title(r"Gaussian kernel with $\gamma={}$".format(gamma), fontsize=14)

# Gaussian kernel 1
landmark1 = np.array([-1.5])
gamma1 = 0.16
if let_plot:
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plt.legend(fontsize=12, loc="upper right")
    plt.show()

# Gaussian kernel 2: larger gamma, more concentrate around the landmark
landmark2 = np.array([0.26])
gamma2 = 0.51
if let_plot:
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=12, loc="upper right")
    plt.show()


# 5.3. Data transformation (>> see slide) 
def plot_transformed_data(X_2D,y,xy_lim=[-4.5, 4.5, -0.1, 1.1]):
    plt.axhline(y=0, color='k') # Ox
    #plt.axvline(x=0, color='k') # Oy
    plt.plot(X_2D[:, 0][y==0], X_2D[:, 1][y==0], "rs", markersize=9, label="Samples (class 0)")
    plt.plot(X_2D[:, 0][y==1], X_2D[:, 1][y==1], "g^", markersize=9, label="Samples (class 1)")

    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=14)
    plt.axis(xy_lim)
    plt.title("Data in new feature space", fontsize=14)

# 5.3.1. Use Gaussian kernel 1 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1)]
    plot_transformed_data(X_2D,y)
    plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()

# 5.3.2. Use Gaussian kernel 2 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark2, gamma2)]
    plot_transformed_data(X_2D,y)
    #plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()


# 5.3.3. Use Gaussian kernels with 2 landmarks (add 2 features)
if let_plot:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plot_kernel(X_1D,y,landmark2,gamma2)    
    plt.title("2 Gaussian kernels", fontsize=14)#plt.legend(fontsize=10, loc="upper right")

    #from mpl_toolkits.mplot3d import Axes3D 
    ax = fig.add_subplot(122, projection='3d')
    X_3D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1), 
                 gaussian_rbf(X_1D, landmark2, gamma2)]
    ax.scatter(X_3D[:, 0][y==0], X_3D[:, 1][y==0], X_3D[:, 2][y==0], 
                s=115,c="red",marker='s',label="Samples (class 0)")
    ax.scatter(X_3D[:, 0][y==1], X_3D[:, 1][y==1], X_3D[:, 2][y==1], 
                s=115,c="green",marker='^',label="Samples (class 1)")

    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$\n(similarity to lm 1)", fontsize=12)
    ax.set_zlabel("$x_3$\n(similarity to lm 2)", fontsize=12)
    plt.title("Data in new feature space", fontsize=14)
    plt.show()

# 5.4. How to choose landmarks? (>> see slide)



# 5.6. Kernel trick for method 2
# Generate non-linear data
X, y = make_moons(n_samples=100, noise=0.22, random_state=42)

# Train 1 Gaussian SVM using Kernel trick 
Gaus_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf",gamma=5, C=0.001))  ])  
Gaus_kernel_svm_clf.fit(X, y)
Gaus_kernel_svm_clf.predict(X)

# Train several Gaussian SVMs using Kernel trick 
gamma1, gamma2 = 0.1, 10
C1, C2 = 0.001, 100
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    Gaus_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("", SVC(kernel="rbf", gamma=gamma, C=C)) ])
    Gaus_kernel_svm_clf.fit(X, y)
    svm_clfs.append(Gaus_kernel_svm_clf)

# Plot boundaries by different SVMs
plt.figure(figsize=(11, 9))
for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(2,2,i+1)
    plot_predictions(svm_clf, [0,20,0,10])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"Use Gaus. kernel with $\gamma = {}, C = {}$".format(gamma, C), fontsize=14)
    if i in (0, 1): 
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")
plt.show()

# 5.7. (>> see slide) What is the effect of: 
#   Large / small C?
#   Large / small gamma: ?


# In[5]: METHOD 2: ADD SIMILARITY FEATURES AND TRAIN 
# 5.1. Generate 1-fearture data (1-dimenstional data)
X_1D = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]).reshape(-1,1) 
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0]) # 2 classes

# 5.2. Plot Gaussian kernel graphs
def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)
def plot_kernel(X_1D,y,landmark,gamma, no_plot_points=200, xy_lim = [-4.5, 4.5, -0.1, 1.1]):  
    # Plot samples:
    plt.axhline(y=0, color='k') # Ox axis
    plt.plot(X_1D[y==0], np.zeros(4), "rs", markersize=9, label="Data samples (class 0)")
    plt.plot(X_1D[y==1], np.zeros(5), "g^", markersize=9, label="Data samples (class 1)")

    # Plot the landmark:
    plt.scatter(landmark, [0], s=200, alpha=0.5, c="orange")
    plt.annotate(r'landmark',xytext=(landmark, 0.2),
                 xy=(landmark, 0), ha="center", fontsize=14,
                 arrowprops=dict(facecolor='black', shrink=0.1)  )
    
    # Plot Gaussian kernel graph: 
    x1_plot = np.linspace(-4.5, 4.5, no_plot_points).reshape(-1,1)  
    x2_plot = gaussian_rbf(x1_plot, landmark, gamma)
    plt.plot(x1_plot, x2_plot, "b--", linewidth=2, label="Gaussian kernel")
    
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=13)
    #plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.axis(xy_lim)
    plt.title(r"Gaussian kernel with $\gamma={}$".format(gamma), fontsize=14)

# Gaussian kernel 1
landmark1 = np.array([-1.5])
gamma1 = 0.16
if let_plot:
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plt.legend(fontsize=12, loc="upper right")
    plt.show()

# Gaussian kernel 2: larger gamma, more concentrate around the landmark
landmark2 = np.array([0.26])
gamma2 = 0.51
if let_plot:
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=12, loc="upper right")
    plt.show()


# 5.3. Data transformation (>> see slide) 
def plot_transformed_data(X_2D,y,xy_lim=[-4.5, 4.5, -0.1, 1.1]):
    plt.axhline(y=0, color='k') # Ox
    #plt.axvline(x=0, color='k') # Oy
    plt.plot(X_2D[:, 0][y==0], X_2D[:, 1][y==0], "rs", markersize=9, label="Samples (class 0)")
    plt.plot(X_2D[:, 0][y==1], X_2D[:, 1][y==1], "g^", markersize=9, label="Samples (class 1)")

    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=14)
    plt.axis(xy_lim)
    plt.title("Data in new feature space", fontsize=14)

# 5.3.1. Use Gaussian kernel 1 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1)]
    plot_transformed_data(X_2D,y)
    plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()

# 5.3.2. Use Gaussian kernel 2 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark2, gamma2)]
    plot_transformed_data(X_2D,y)
    #plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()


# 5.3.3. Use Gaussian kernels with 2 landmarks (add 2 features)
if let_plot:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plot_kernel(X_1D,y,landmark2,gamma2)    
    plt.title("2 Gaussian kernels", fontsize=14)#plt.legend(fontsize=10, loc="upper right")

    #from mpl_toolkits.mplot3d import Axes3D 
    ax = fig.add_subplot(122, projection='3d')
    X_3D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1), 
                 gaussian_rbf(X_1D, landmark2, gamma2)]
    ax.scatter(X_3D[:, 0][y==0], X_3D[:, 1][y==0], X_3D[:, 2][y==0], 
                s=115,c="red",marker='s',label="Samples (class 0)")
    ax.scatter(X_3D[:, 0][y==1], X_3D[:, 1][y==1], X_3D[:, 2][y==1], 
                s=115,c="green",marker='^',label="Samples (class 1)")

    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$\n(similarity to lm 1)", fontsize=12)
    ax.set_zlabel("$x_3$\n(similarity to lm 2)", fontsize=12)
    plt.title("Data in new feature space", fontsize=14)
    plt.show()

# 5.4. How to choose landmarks? (>> see slide)



# 5.6. Kernel trick for method 2
# Generate non-linear data
X, y = make_moons(n_samples=100, noise=0.22, random_state=42)

# Train 1 Gaussian SVM using Kernel trick 
Gaus_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf",gamma=5, C=0.001))  ])  
Gaus_kernel_svm_clf.fit(X, y)
Gaus_kernel_svm_clf.predict(X)

# Train several Gaussian SVMs using Kernel trick 
gamma1, gamma2 = 0.1, 10
C1, C2 = 0.001, 100
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    Gaus_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("", SVC(kernel="rbf", gamma=gamma, C=C)) ])
    Gaus_kernel_svm_clf.fit(X, y)
    svm_clfs.append(Gaus_kernel_svm_clf)

# Plot boundaries by different SVMs
plt.figure(figsize=(11, 9))
for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(2,2,i+1)
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"Use Gaus. kernel with $\gamma = {}, C = {}$".format(gamma, C), fontsize=14)
    if i in (0, 1): 
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")
plt.show()

# 5.7. (>> see slide) What is the effect of: 
#   Large / small C?
#   Large / small gamma: ?


'''________________________________________________'''
# Câu 3: Train polynomial SVM regression trong mục 6.5 sử dụng hyperparameters 
# degree>2 và các giá trị khác nhau cho C và epsilon. 
# Giải thích kết quả thu được.


# In[6]: SVM REGRESSION
# 6.1. Generata non-linear 1D data
np.random.seed(42)
m = 200 
X = 4*np.random.rand(m, 1) -2
y = (4 + 3*X**3 + X + np.random.randn(m, 1)).ravel()

# 6.2. Fit Linear SVM regressors
from sklearn.svm import LinearSVR
svm_reg1 = LinearSVR(epsilon=2, random_state=42)
svm_reg1.fit(X, y)
svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)      
svm_reg2.fit(X, y)

# 6.3. Plot the hypothesis
def plot_svm_regression(svm_reg, X, y, axes):
    # Plot model, margins
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=3, label=r"Hypothesis $\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "b--", linewidth=1, label="Margins")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "b--", linewidth=1)
    
    # Mask violated samples:
    #plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    # Plot samples:
    plt.plot(X, y, "bo")
    
    plt.axis(axes)

# 6.4. Which one fits the data better? (>> see slide)


# 6.5. Non-linear SVM regression
from sklearn.svm import SVR
# Recall: 
#   smaller epsilon ==> less data fitted
#   smaller C ==> less overfitting
svm_poly_reg1 = SVR(kernel="poly", degree=3, epsilon=0.02, C=0.001, gamma="scale")
svm_poly_reg1.fit(X, y)
svm_poly_reg2 = SVR(kernel="poly", degree=3, epsilon=0.2, C=2, gamma="scale")
svm_poly_reg2.fit(X, y)
svm_poly_reg3 = SVR(kernel="poly", degree=3, epsilon=2, C=0.001, gamma="scale")
svm_poly_reg3.fit(X, y)
svm_poly_reg4 = SVR(kernel="poly", degree=3, epsilon=2, C=2, gamma="scale")
svm_poly_reg4.fit(X, y)

if let_plot:
    plt.figure(figsize=(12, 9))
    plt.subplot(2,2,1)
    xylim = [-2, 2, 3, 11]
    plot_svm_regression(svm_poly_reg1, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg1.degree, svm_poly_reg1.epsilon, svm_poly_reg1.C), fontsize=14)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    
    plt.subplot(2,2,2)
    plot_svm_regression(svm_poly_reg2, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg2.degree, svm_poly_reg2.epsilon, svm_poly_reg2.C), fontsize=14)
    
    plt.subplot(2,2,3)
    plot_svm_regression(svm_poly_reg3, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg3.degree, svm_poly_reg3.epsilon, svm_poly_reg3.C), fontsize=14)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.xlabel(r"$x_1$", fontsize=14)
    
    plt.subplot(2,2,4)
    plot_svm_regression(svm_poly_reg4, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg4.degree, svm_poly_reg4.epsilon, svm_poly_reg4.C), fontsize=14)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.show()

# 6.6. (exercise) Explain why epsilon=1 leads to wrong models (with both large and small C).
print("\n")



# %%