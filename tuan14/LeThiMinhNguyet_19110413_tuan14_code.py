# In[0]: IMPORTS, SETTINGS 
import sys
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import numpy as np
import os
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
let_plot = False


# ===== VOTING METHODS =====
print('\n')


# In[1]: VOTING CLASSIFIERS
# 1.1. Hard voting
# 1.1.1. Generate data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 1.1.2. Create 3 diverse classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 1.1.3. Combine them to create VotingClassifier
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard') # Notice: voting='hard'

# 1.1.4. Train models
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)

# 1.1.5. Try prediction
print("====== HARD voting ======")
samples = np.concatenate((X_test[y_test==0][14:15], X_test[y_test==1][14:15]))
samples_lbl = np.concatenate((y_test[y_test==0][14:15], y_test[y_test==1][14:15]))
for sample, label in zip(samples, samples_lbl):
    print("\nPrediction of classifiers:")
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        print("\t",clf.__class__.__name__,": predicted class ", clf.predict([sample]))
    print("True label: ",label)

# 1.1.6. Compute their accuracy
from sklearn.metrics import accuracy_score
print("Performance of 3 diverse models and a HARD voting classifier:")
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\t",clf.__class__.__name__, accuracy)
print("\n")
# Observation: Better than the best?

# 1.2. Soft voting (>> see slide)
# 1.2.1. Create and train a soft voting classifier
voting_clf_soft = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft') # Notice: voting='hard' 
voting_clf_soft.fit(X_train, y_train) 

# 1.2.2. Try prediction
print("====== SOFT voting ======")
samples = np.concatenate((X_test[y_test==0][14:15], X_test[y_test==1][14:15]))
samples_lbl = np.concatenate((y_test[y_test==0][14:15], y_test[y_test==1][14:15]))
for sample, label in zip(samples, samples_lbl):
    print("\nPrediction of classifiers:")
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf_soft):
        if clf.__class__.__name__ != "VotingClassifier":
            print("\t",clf.__class__.__name__,
                  ", class proba: ", np.round(clf.predict_proba([sample]),2))
        else:
            print("\t",clf.__class__.__name__,
                  ", class proba: ", np.round(clf.predict_proba([sample]),2),
                  " ==> predicted class ", clf.predict([sample]))
    print("True label: ",label)

# 1.2.3. Compute their accuracy      
print("Performance of 3 diverse models and a SOFT voting classifier:")
for clf in (log_clf, rnd_clf, svm_clf, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\t",clf.__class__.__name__, accuracy_score(y_test, y_pred))
print("\n")


# In[2]: RANDOM SAMPLING (>> see slide)
# 2.1. Generate data
from sklearn.datasets import make_moons
let_plot=True
m = 500
X, y = make_moons(n_samples=m, noise=0.3, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2.2. Train a decision tree (no restrictions on max_depth,...)
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
from sklearn.metrics import accuracy_score
tree_acc = accuracy_score(y_test, y_pred_tree) 
print("Decision tree accuracy: ", tree_acc)


# 2.3. Train an ensemble model of decision trees
from sklearn.ensemble import BaggingClassifier
ens_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), 
    n_estimators=100, # no. of predictors (decision trees in this case)
    bootstrap=False, 
    max_samples=200, # no. of samples in each training sets 
    n_jobs=3,   # no. of CPU cores used to train (-1: use all)
    random_state=42)
# Note: BaggingClassifier uses SOFT voting (if estimator has predict_proba()) 
ens_clf.fit(X_train, y_train)
y_pred = ens_clf.predict(X_test)
ens_acc = accuracy_score(y_test, y_pred) 
print("Ensemble model's accuracy: ", ens_acc)


from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.9, contour=True, plot_samples = True):
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.9)
    if plot_samples:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "ro", alpha=alpha)
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)
if let_plot:
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plot_decision_boundary(tree_clf, X, y)
    plt.title("Decision tree (with no restrictions). \nTest acc. = {}".format(tree_acc), fontsize=14)
    plt.subplot(122)
    plot_decision_boundary(ens_clf, X, y)
    plt.title("Ensemble of {} decision trees \n(trained with bootstrap={}). Test acc. = {}".format(ens_clf.n_estimators,ens_clf.bootstrap,ens_acc), fontsize=14)
    #plt.savefig("figs/decision_tree_vs_ensemble_model")
    plt.show()

# Ở hình 1 decision tree không có giới hạn nên cố gắn chia ra nhiều node để đi xen qua các dữ liệu nên bị overfitting
# Ở hình 2 thì model của 100 cây hợp lại, model khá đẹp phân chia được ra 2 phân dữ liệu do đã có các giới hạn cho model
# Độ chính xác cũng cao hơn so với model 1 với 88%, model 1 chỉ có 86%



# 2.5. Use out-of-bag evaluation 
ens_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), 
    oob_score = True,
    n_estimators=100, max_samples=300, bootstrap=True, 
    n_jobs=3, random_state=42)
ens_clf.fit(X_train, y_train)
y_pred = ens_clf.predict(X_train)
ens_acc = accuracy_score(y_train, y_pred) 
print("Ensemble model's accuracy (on training set): ", ens_acc)
print("Ensemble model's accuracy (on oob samples): ", ens_clf.oob_score_)
y_pred = ens_clf.predict(X_test)
ens_acc = accuracy_score(y_test, y_pred) 
print("Ensemble model's accuracy (on validation set): ", ens_acc)


# In[3]: RANDOM FORESTS
# 3.1. Generate data
from sklearn.datasets import make_moons
m = 500
X, y = make_moons(n_samples=m, noise=0.3, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

# 3.2. Way 1: Implement using BaggingClassifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_leaf_nodes=16, random_state=42), # splitter="random", 
    n_estimators=200, max_samples=.7, bootstrap=True, random_state=42 )
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# 3.3. Way 2: Implement using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_leaf_nodes=16, 
                                 n_estimators=200, 
                                 max_samples=.7, 
                                 bootstrap=True, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# 3.4. Count no. of identical predictions
len(X_test) # no. of test samples
np.sum(y_pred == y_pred_rf) # almost the same predictions

# 3.5. Compute accuracies
accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred_rf)


# 3.6. Try models on Iris data
# Load Iris data 
from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, 
    test_size=0.3, random_state=42)
 
# Train models
bag_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# Compute accuracies
y_pred = bag_clf.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)
accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred_rf)


# 3.7. Feature importance
rf_clf.feature_importances_ 
sum(rf_clf.feature_importances_)


# In[4]: EXTREMELY RANDOMIZED TREES (EXTRA-TREES)
# 4.1. Try on moon data
m = 500
X, y = make_moons(n_samples=m, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
from sklearn.ensemble import ExtraTreesClassifier
extra_trees = ExtraTreesClassifier(max_leaf_nodes=16, 
                                 n_estimators=200, 
                                 max_samples=.7, 
                                 bootstrap=True, random_state=42)
# Train and test
extra_trees.fit(X_train, y_train)
accuracy_score(y_test, extra_trees.predict(X_test))
# Compare with normal random forest
rf_clf.fit(X_train, y_train)
accuracy_score(y_test, rf_clf.predict(X_test))


# ===== BOOSTING METHODS =====
print('\n')


# In[5]: ADABOOST 
# 5.1. Create data
m = 500
X, y = make_moons(n_samples=m, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
                     
# 5.2. Implement AdaBoost manually to see individual predictors (decision tree in this case)
sample_weights = np.ones(len(X_train))
weight_boost_rate = 2 ##### INCREASE this to add MORE weights to misclassifications

plt.figure(figsize=(22,5))
n_predictor_plot=3
for iter in range(1,n_predictor_plot+1):
    # Train a predictor
    svm_clf = SVC(C=0.05, random_state=42) # kernel="rbf", gamma="scale", 
    svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Update weights: larger weights on misclassifications
    y_pred = svm_clf.predict(X_train)
    sample_weights[y_pred != y_train] *= weight_boost_rate
    
    if let_plot:
        plt.subplot(1,n_predictor_plot,iter)

        # Plot samples (larger weights, higher alpha)                 
        # Class 0
        no_of_samples = len(X_train[y_train==0])
        weight_of_samples = sample_weights[y_train==0]
        rgba_colors = np.zeros((no_of_samples,4))
        rgba_colors[:,0] = 1.0 # 1st col: red
        rgba_colors[:, 3] = weight_of_samples/max(weight_of_samples) # last col: alpha
        plt.scatter(X_train[y_train==0][:,0],X_train[y_train==0][:,1],marker='o',color=rgba_colors)
        # Class 1
        no_of_samples = len(X_train[y_train==1])
        weight_of_samples = sample_weights[y_train==1]
        rgba_colors = np.zeros((no_of_samples,4))
        rgba_colors[:,2] = 1.0 # blue
        rgba_colors[:, 3] = weight_of_samples/max(weight_of_samples) # last col: alpha
        plt.scatter(X_train[y_train==1][:,0],X_train[y_train==1][:,1],marker='s',color=rgba_colors)

        # Plot decision boundary
        plot_decision_boundary(svm_clf, X, y, alpha=0.2,plot_samples = False)
    
        plt.title("Predictor {}".format(iter))
        if iter!=1: plt.ylabel("")
plt.savefig("figs/AdaBoost_predictors")
plt.show()

# 5.3.Train AdaBoost classifier
from sklearn.ensemble import AdaBoostClassifier
# Note: MUST set probability=True in SVC() to use with AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    SVC(probability=True, C=0.05, random_state=42), algorithm='SAMME.R', #DecisionTreeClassifier(), 
    n_estimators=10, random_state=42)
ada_clf.fit(X_train, y_train)
if let_plot:
    plot_decision_boundary(ada_clf, X, y)
    #plt.savefig("figs/AdaBoost_final")
    plt.show()





# In[6]: GRADIENT BOOSTING 
# 6.1. Generate data
X = np.random.rand(100, 1) - 0.5
#y = 3*X[:, 0]**2 + 0.05*np.random.randn(100,1)
y = 3*X**2 + 0.05*np.random.randn(100,1); y = y.ravel() 

# 6.2. Manually train 3 trees sequentially. The latter fits residual errors of the former. 
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X) # residual errors of tree1
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X) # residual errors of tree2
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

# 6.3. Try prediction (sum of predictions by trees) 
X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
y_pred

# 6.4. Plot data and hypotheses of trees 
def plot_grad_boost_hypothesis(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    # Plot samples
    plt.plot(X[:, 0], y, data_style, label=data_label)
    
    # Plot hypotheses (predictions)
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    
    if label or data_label: plt.legend(loc="upper right", fontsize=14)
    plt.axis(axes)

if let_plot: 
    # To prevent Windows 10 scaling "Change the size of apps and text"
    from ctypes import *
    windll.shcore.SetProcessDpiAwareness(c_int(1))

    # Plot samples and hypothesis of tree1
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plot_grad_boost_hypothesis([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], label="$h_1(x_1)$", style="r-", data_style="b.", data_label="Training sample")
    plt.ylabel("$y$", fontsize=14, rotation=0)
    plt.xlabel("$x_1$", fontsize=14)
    plt.title("Training samples \nand hypothesis of tree1", fontsize=16)

    # Plot residuals and hypothesis of tree2 
    plt.subplot(132)
    plot_grad_boost_hypothesis([tree_reg2], X, y2, axes=[-0.5, 0.5,-0.2, 0.8], label="$h_2(x_1)$", style="r-", data_style="b^", data_label="Error")
    plt.ylabel("$y - h_1(x_1)$", fontsize=14)
    plt.xlabel("$x_1$", fontsize=14)
    plt.title("Tree1's residual errors \nand hypothesis of tree2", fontsize=16)

    # Plot residuals and hypothesis of tree3 
    plt.subplot(133)
    plot_grad_boost_hypothesis([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.2, 0.8], label="$h_3(x_1)$", style="r-", data_style="b^", data_label="Error")
    plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=14)
    plt.xlabel("$x_1$", fontsize=14)
    plt.title("Tree2's residual errors \nand hypothesis of tree2", fontsize=16)
    plt.tight_layout()
    plt.savefig("figs/gradient_boosting_training")
    
    # Plot hypothesis (predictions) of tree1
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plot_grad_boost_hypothesis([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], label="$h(x_1) = h_1(x_1)$", style="r-", data_style="b.")
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$y$", fontsize=12, rotation=0)
    plt.title("Predictions of tree1", fontsize=16)
    
    # Plot hypothesis by (sum predictions of) 2 trees 
    plt.subplot(132)
    plot_grad_boost_hypothesis([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.2, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$", style="r-", data_style="b.")
    plt.xlabel("$x_1$", fontsize=12)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)
    plt.title("Total predictions of 2 trees", fontsize=16)
    
    # Plot hypothesis by (sum predictions of) 3 trees 
    plt.subplot(133)
    plot_grad_boost_hypothesis([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.2, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$", style="r-", data_style="b.")
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)
    plt.title("Total predictions of 3 trees", fontsize=16)       
    plt.tight_layout()
    plt.savefig("figs/gradient_boosting_hypotheses")
    plt.show()

# (>> see slide)

# 6.5. Implement gradient boosting with sklearn
# Note: Decision trees are the base predictors of GradientBoostingRegressor 
from sklearn.ensemble import GradientBoostingRegressor

# Note: In GradientBoostingRegressor, learning_rate is a regularization parameter.
#   It shrinks the contribution of each tree.  
#   => Smaller LEARNING_RATE, LESS overfitting (i.e., better generalization)
#   => Smaller LEARNING_RATE, require MORE trees to fit data
gbrt1 = GradientBoostingRegressor(max_depth=2, n_estimators=5, learning_rate=1, random_state=42)
gbrt1.fit(X, y)
gbrt2 = GradientBoostingRegressor(max_depth=2, n_estimators=5, learning_rate=0.06, random_state=42)
gbrt2.fit(X, y)
gbrt3 = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=1, random_state=42)
gbrt3.fit(X, y)                                                                  
gbrt4 = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=0.06, random_state=42)
gbrt4.fit(X, y)

# Plot hyprotheses of trained models
if let_plot:
    plt.figure(figsize=(11,11))

    plt.subplot(221)
    plot_grad_boost_hypothesis([gbrt1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
    plt.title("n_estimators={}, learning_rate={}".format(gbrt1.n_estimators, gbrt1.learning_rate), fontsize=14)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)

    plt.subplot(222)
    plot_grad_boost_hypothesis([gbrt2], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("n_estimators={}, learning_rate={}".format(gbrt2.n_estimators, gbrt2.learning_rate), fontsize=14)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)

    plt.subplot(223)
    plot_grad_boost_hypothesis([gbrt3], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("n_estimators={}, learning_rate={}".format(gbrt3.n_estimators, gbrt3.learning_rate), fontsize=14)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)

    plt.subplot(224)
    plot_grad_boost_hypothesis([gbrt4], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("n_estimators={}, learning_rate={}".format(gbrt4.n_estimators, gbrt4.learning_rate), fontsize=14)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)
    plt.tight_layout()
    plt.savefig("figs/gbrt_learning_rate")
    plt.show()


# 6.6. Find no. of trees using Early stopping methods
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 6.6.1. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

# 6.6.2. Train boosting models
gbrt = GradientBoostingRegressor(learning_rate=0.1,max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

# n_estimators lớn model sẽ bị underfitting để cân bằng lại em sẽ giảm learning_rate


# 6.6.3. Compute errors (in validatation set) w.r.t. no. of tree
# Note: staged_predict() returns predictions at each stage of training (with 1 tree, 2 trees, etc.)
errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]

# 6.6.4. Get the best model:
best_no_of_trees = np.argmin(errors) + 1
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_no_of_trees, random_state=42)
gbrt_best.fit(X_train, y_train)

# 6.6.5. Plot learning curve and hypothesis of the best model
if let_plot: 
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(errors, "b.-")
    min_error = np.min(errors)  
    plt.plot([best_no_of_trees, best_no_of_trees], [0, min_error], "k--")
    plt.plot([0, 120], [min_error, min_error], "k--")
    plt.plot(best_no_of_trees, min_error, "ko")
    plt.text(best_no_of_trees, min_error*1.2, "Minimum", ha="center", fontsize=14)
    plt.axis([0, 120, 0, 0.01])
    plt.xlabel("Number of trees")
    plt.ylabel("Validation error", fontsize=14)
    plt.title("Learning curve", fontsize=14)
    # Plot hypothesis of the best model
    plt.subplot(122)
    plot_grad_boost_hypothesis([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("Hypothesis of the best model (%d trees)" % best_no_of_trees, fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)
    plt.xlabel("$x_1$", fontsize=14)
    #plt.savefig("figs/early_stopping_gbrt")
    plt.show()

# Thấy rằng ở lần chạy đầu tiên với số lỗi khá cao
# Và giá trị minimum nằm ở 56 cây
# Với model dùng 56 tree thì thấy model tốt khống bị underfitting hay là overfitting


# In[7]: XGBoost
# 7.1. Train and test an XGBoost model
# Documentation: https://xgboost.readthedocs.io/en/latest/python/python_api.html
import xgboost
xgb_reg = xgboost.XGBRegressor(random_state=42)
xgb_reg.n_estimators # number of trees. Default is 100 trees.
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val) # use ALL trees (n_estimators) to predict
val_error = np.sqrt(mean_squared_error(y_val, y_pred))  
val_error         

# 7.2. Find best no. of trees using early stopping with XGBoost lib
# Training note: 
#   early_stopping_rounds=5: stop when no improvement after adding 5 trees
xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=True)
# Prediction notes: 
#   .best_ntree_limit: optimal no. of trees
#   .predict(ntree_limit=NN): NN is the no. of trees to use in prediction (default is 0: use all learned trees)
xgb_reg.best_ntree_limit  
y_pred = xgb_reg.predict(X_val, ntree_limit=xgb_reg.best_ntree_limit) 
#y_pred = xgb_reg.predict(X_val, ntree_limit=100) # this prediction is the same as 7.1's (uses 100 trees) 
val_error = np.sqrt(mean_squared_error(y_val, y_pred))   
val_error          

DONE = True