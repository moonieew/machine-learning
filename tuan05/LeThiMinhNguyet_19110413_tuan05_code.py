'''
The following code is mainly from Chap 3, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb

LAST REVIEW: Nov 2020
'''


# In[0]: IMPORTS
from types import MethodType
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib # Note: require sklearn v0.22+ (to update sklearn: pip install -U scikit-learn ). For old version sklearn: from sklearn.externals import joblib 


# In[1]: MNIST DATASET
# 1.1. Load MNIST  
# load dữ liệu của bộ mnist, nó sẽ chia ra tập train và tập test    
from tensorflow import keras
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#X_train  : xem tập train
#X_train.shape :xem kích thước của nó: 60000 ngàn tấm hình kích thước 28x28

# 1.2. Reshape to 2D array: each row has 784 features
#reshape là đổi nó lại thành mảng 1 chiều 28x28 =784 là đổi thành 1 cái bảng có 60000 dòng và 784 cột (784 feature)
train_images=train_images.reshape(60000,784)
test_images=test_images.reshape(10000,784)

# 1.3. Plot a digit image  
# hàm plot ra 1 tấm hình 
def plot_clothes(data, label = 'unspecified', showed=True):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("Clothes: " + str(label))
    #plt.axis("off")
    if showed:
        plt.show()
#plot dòng dữ liệu 0 thì ra hình số 5
sample_id = 0
plot_clothes(train_images[sample_id], train_labels[sample_id])


# In[2]: TRAINING A BINARY CLASSIFIER (just two classes, 5 and not-5)               
# 2.1. Create label array: True for 5s, False for other digits.
#huấn luyện BINARY CLASSIFIER (2 lớp)

#chỗ nào số 5 là true không phải là false
y_train_5 = (y_train == 5) 
y_test_5 = (y_test == 5)
#xem y_train và y_train_5 để thấy nào true nào false

# 2.2. Try Stochastic Gradient Descent (SGD) classifier
# Note 1: In sklearn, SGDClassifier train linear classifiers using SGD, 
#         depending on the loss, eg. ‘hinge’ (default): linear SVM, ‘log’: logistic regression, etc.
# Note 2: SGD takes 1 datum at a time, hence well suited for online learning. 
#         It also able to handle very large datasets efficiently
#dùng SGDClassifier hiểu là linear model
from sklearn.linear_model import SGDClassifier       
sgd_clf = SGDClassifier(random_state=42) # set random_state to reproduce the result
# Train: 
# Warning: takes time for new run!
#training dữ liệu, nếu muốn train thì bật new_run= true
new_run = False
if new_run == True:
    sgd_clf.fit(X_train, y_train_5)
    joblib.dump(sgd_clf,'saved_var/sgd_clf_binary')
else:
    sgd_clf = joblib.load('saved_var/sgd_clf_binary')

# Predict a sample:
#dự đoán thử 1 hình ảnh
sample_id = 4
plot_digit(X_train[sample_id], label=y_train[sample_id])
#dự đoán của model của mình ra kq = false là dự đoán đúng
sgd_clf.predict([X_train[sample_id]])
y_train_5[sample_id]


# In[3]: PERFORMANCE MEASURES 
#đo khả năng thực thi của nó
# 3.1. Accuracy (with cross-validation) of SGDClassifier 
from sklearn.model_selection import cross_val_score
# Warning: takes time for new run! 
# đo độ chính xác của nó, cv=3 là chạy 3 lần
if new_run == True:
    accuracies = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    joblib.dump(accuracies,'saved_var/sgd_clf_binary_acc')
else:
    accuracies = joblib.load('saved_var/sgd_clf_binary_acc')
#xem độ chính xác: accuracies, chạy 3 lần nên ra 3 con số: 95, 96%

# 3.2. Accuracy of a dump classifier
# Note: We are having an IMBALANCED data, hence accuracy is not useful!
from sklearn.base import BaseEstimator

class DumpClassifier(BaseEstimator): # always return False (not-5 label)
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
no_5_model = DumpClassifier()
cross_val_score(no_5_model, X_train, y_train_5, cv=3, scoring="accuracy")
# Note: 
#   >90% accuracy, due to only about 10% of the images are 5s.
#   IMBALANCED (or skewed) datasets: some classes are much more frequent than others.

# 3.3. Confusion matrix (better for imbalanced data)
# Info: number of times the classifier "confused" b/w samples of classes
from sklearn.model_selection import cross_val_predict 
# Warning: takes time for new run! 
if new_run == True:
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    joblib.dump(sgd_clf,'saved_var/y_train_pred')
else:
    y_train_pred = joblib.load('saved_var/y_train_pred')

from sklearn.metrics import confusion_matrix  
conf_mat = confusion_matrix(y_train_5, y_train_pred) # row: actual class, column: predicted class. 
# Perfect prediction: zeros off the main diagonal 
y_train_perfect_predictions = y_train_5  # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)

# 3.4. Precision and recall (>> see slide)
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

# 3.5. F1-score 
# Info: F1-score is the harmonic mean of precision and recall. 1: best, 0: worst.
#   F1 = 2 × precision × recall / (precision + recall)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

# 3.6. Precision/Recall tradeoff (>> see slide) 
# 3.6.1. Try classifying using some threshold (on score computed by the model)  
sample_id = 11
y_score = sgd_clf.decision_function([X_train[sample_id]]) # score by the model
threshold = 0
y_some_digit_pred = (y_score > threshold)
y_train_5[sample_id]
# Raising the threshold decreases recall
threshold = 10000
y_some_digit_pred = (y_score > threshold)  

# 3.6.2. Precision, recall curves wrt to thresholds 
# Get scores of all intances
# Warning: takes time for new run! 
if new_run == True:
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    joblib.dump(sgd_clf,'saved_var/y_scores')
else:
    y_scores = joblib.load('saved_var/y_scores')

# Plot precision,  recall curves
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
let_plot = False
if let_plot:
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend() 
    plt.grid(True)
    plt.xlabel("Threshold")   

# Plot a threshold
thres_value = 1000
thres_id = np.min(np.where(thresholds >= thres_value))
precision_at_thres_id = precisions[thres_id] 
recall_at_thres_id = recalls[thres_id] 
if let_plot:
    plt.plot([thres_value, thres_value], [0, precision_at_thres_id], "r:")    
    plt.plot([thres_value], [precision_at_thres_id], "ro")                            
    plt.plot([thres_value], [recall_at_thres_id], "ro")            
    plt.text(thres_value+500, 0, thres_value)    
    plt.text(thres_value+500, precision_at_thres_id, np.round(precision_at_thres_id,3))                            
    plt.text(thres_value+500, recall_at_thres_id, np.round(recall_at_thres_id,3))     
    plt.show()

# 3.6.3. Precision vs recall curve (Precision-recall curve)
if let_plot:         
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.title("Precision-recall curve (PR curve)")
    plt.show()


# 3.7. Receiver operating characteristic (ROC) curve 
# Info: another common measure for binary classifiers. 
# ROC curve: the True Positive Rate (= recall) against 
#   the False Positive Rate (= no. of false postives / total no. of actual negatives).
#   FPR is the ratio of negative instances that are incorrectly classified as positive.
# NOTE: 
#   Tradeoff: the higher TPR, the more FPR the classifier produces.
#   Good classifier goes toward the top-left corner.

# 3.7.1. Compute FPR, TPR for the SGDClassifier
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# 3.7.2. Compute FPR, TPR for a random classifier (make prediction randomly)
from sklearn.dummy import DummyClassifier
dmy_clf = DummyClassifier(strategy="uniform")
y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_dmy = y_probas_dmy[:, 1]
fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)

# 3.7.3. Plot ROC curves
if let_plot:
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot(fprr, tprr, 'k--') # random classifier
    #plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal: random classifier
    plt.legend(['SGDClassifier','Random classifier'])
    plt.grid(True)        
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate (Recall)')    
    plt.show()

# 3.8. Compute Area under the curve (AUC) for ROC
# Info: 
#   A random classifier: ROC AUC = 0.5.
#   A perfect classifier: ROC AUC = 1.
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_train_5, y_scores)
    
# 3.9. ROC vs PR curve: when to use?
#   PR curve: focus the false positives (ie. u want high precision)
#   ROC: focus the false negatives (ie. u want high recall)
print('\n')
