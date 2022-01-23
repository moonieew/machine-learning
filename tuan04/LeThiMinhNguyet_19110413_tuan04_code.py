'''
The following code is mainly from Chap 2, Géron 2019
See https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb

LAST REVIEW: Oct 2020
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# np.set_printoptions(threshold = np.inf)
# pd.options.display.max_columns = 20
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from statistics import mean


# In[1]: LOOK AT THE BIG PICTURE (DONE)
raw_data = pd.read_csv('LeThiMinhNguyet_19110413_tuan04_giaxeoto.csv') #hàm đọc dữ liệu từ file 


# In[3]: DISCOVER THE DATA TO GAIN INSIGHTS
# một số hàm trong padas
 #đọc thông tin bao nhiêu cột, mô tả tóm tắt mỗi cột(mỗi cột có bao nhiêu dữ liệu null), loại dữ liệu của mỗi cột
print('\n____________________________________ Dataset info ____________________________________')
print(raw_data.info()) 
#lấy 7 dòng đầu để xem tương ứng với 7 dòng trong exel
print('\n____________________________________ Some first data examples ____________________________________')
print(raw_data.head(7))  
# đếm dữ liệu trong cột DÒNG XE
print('\n____________________________________ Counts on a feature ____________________________________')
print(raw_data['DÒNG XE'].value_counts())
#mô tả dữ liệu: count là đếm giá trị tương đương với non-null ở trên, mean là giá trị trung bình của từng cột, std là phân phối chuẩn có giá trị càng nhỏ thì giá trị nằm xung quanh giá trị trung bình
#có giả trị min, max 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(raw_data.describe())
#truy xuất dòng dữ liệu
print('\n____________________________________ Get specific rows and cols ____________________________________')
#lấy theo tên GIÁ với cột HÃNG XE với hàng 0 7 8
print(raw_data.loc[[0,7,8], ['GIÁ', 'HÃNG XE']] ) # Refer using column name
#lấy theo số, lấy cột 2 với 7 với hàng 0 6 9
print(raw_data.iloc[[0,6,9], [2, 7]] ) # Refer using column ID

# 3.2 Scatter plot b/w 2 features
#matplotlib inline
# vẽ biểu đồ kiểu scatter với Giá theo Số chỗ ngồi cột x là SỐ CHỖ NGỒI và Y là GÍA, biểu đồ được lưu trong thư mục figures, những điểm mờ nhạt ở xa xa là những giá trị bất thường
if 0:
    raw_data.plot(kind="scatter", y="GIÁ", x="SỐ CHỖ NGỒI", alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()
# vẽ biểu đồ GIÁ theo NĂM SẢN XUẤT
if 0:
    raw_data.plot(kind="scatter", y="GIÁ", x="NĂM SẢN XUẤT", alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    #plt.savefig('figures/scatter_2_feat.png', format='png', dpi=300)
    plt.show()
#%% 3.3 Scatter plot b/w every pair of features
#vẽ hết tất cả các cột dữ liệu số
if 0:
    from pandas.plotting import scatter_matrix
    features_to_plot = ["GIÁ", "SỐ CHỖ NGỒI", "SỐ KM ĐÃ ĐI", "NĂM SẢN XUẤT"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal vì vẽ đường 45 độ sẽ vô nghĩa
    plt.savefig('figures/scatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

# 3.4 Plot histogram of 1 feature
# vẽ 1 histogram riêng, vẽ cột SỐ CHỖ NGỒI (chia ra nhiều khoảng và đếm xem có bao nhiêu feature trong đó)
if 0:
    from pandas.plotting import scatter_matrix
    features_to_plot = ["SỐ CHỖ NGỒI"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.show()

# 3.5 Plot histogram of numeric features
#vẽ tất cả các histogram của dữ liệu
if 0:
    #raw_data.hist(bins=10, figsize=(10,5)) #bins: no. of intervals
    raw_data.hist(figsize=(10,5)) #bins: no. of intervals
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
    plt.show()

# 3.6 Compute correlations b/w features
#xem độ tương quan của giá trị, cột có độ tương quan giữa 2 giá trị giống nhau là =1, độ tương quan càng gần 1 thì giá trị sẽ tăng cùng nhau,
# càng gầng -1 thì sẽ giảm cùng nhau 
corr_matrix = raw_data.corr()
print(corr_matrix) # print correlation matrix
print(corr_matrix["GIÁ"].sort_values(ascending=False)) # print correlation b/w a feature and other features

# 3.7 Try combining features
#tạo ra cột mới rồi tính lại độ tương quan xem có thay đổi không
raw_data["DIỆN TÍCH PHÒNG"] = raw_data["DIỆN TÍCH - M2"] / raw_data["SỐ PHÒNG"]
raw_data["TỔNG SỐ PHÒNG"] = raw_data["SỐ PHÒNG"] + raw_data["SỐ TOILETS"]
corr_matrix = raw_data.corr()
print(corr_matrix["GIÁ - TRIỆU ĐỒNG"].sort_values(ascending=False)) # print correlation b/w a feature and other features
#xóa 2 cột mới để không ảnh hưởng đến giá trị của dữ liệu
raw_data.drop(columns = ["DIỆN TÍCH PHÒNG", "TỔNG SỐ PHÒNG"], inplace=True) # remove experiment columns


# In[04]: PREPARE THE DATA
# 4.1 Remove unused features
#loại bỏ những cột không dùng
raw_data.drop(columns = ["STT", "SẢN PHẨM", "NHU CẦU", "TỈNH THÀNH",
                         "QUẬN HUYỆN", "PHƯỜNG/ XÃ", "ĐƯỜNG, KHU VỰC","NGÀY ĐĂNG", "DUNG TÍCH", "MÀU XE", "TRỌNG TẢI"], inplace=True)

# 4.2 Split training-test set and NEVER touch test set until test phase
#cắt dữ liệu thành 2 khúc tran và test và không được dùng phần dữ liệu test
#lấy 20% dữ liệu để test, dữ liệu lấy ngẫu nhiên
method = 2
if method == 1: # Method 1: Randomly select 20% of data for test set. Used when data set is large
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42) # set random_state to get the same training set all the time,
#cắt theo Stratified sampling, cột train và test sẽ có hình dạng giống nhau                                                                                      # otherwise, when repeating training many times, your model may see all the data
elif method == 2: # Method 2: Stratified sampling, to remain distributions of important features, see (Geron, 2019) page 56
    # Create new feature "KHOẢNG GIÁ": the distribution we want to remain
    raw_data["KHOẢNG GIÁ"] = pd.cut(raw_data["GIÁ"],
                                    bins=[0, 2000, 4000, 6000, 8000, 12000, 15000, np.inf],
                                    #labels=["<2 tỷ", "2-4 tỷ", "4-6 tỷ", "6-8 tỷ", "8-10 tỷ", ">10 tỷ"])
                                    labels=[2,4,6,8,100,120,150]) # use numeric labels to plot histogram

    # Create training and test set
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
    for train_index, test_index in splitter.split(raw_data, raw_data["KHOẢNG GIÁ"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]

    # See if it worked as expected
    if 1:
        raw_data["KHOẢNG GIÁ"].hist(bins=6, figsize=(5,5)); #plt.show();
        test_set["KHOẢNG GIÁ"].hist(bins=6, figsize=(5,5)); plt.show()

    # Remove the new feature
    print(train_set.info())
    for _set_ in (train_set, test_set):
        #_set_.drop("income_cat", axis=1, inplace=True) # axis=1: drop cols, axis=0: drop rows
        _set_.drop(columns="KHOẢNG GIÁ", inplace=True)
    print(train_set.info())
    print(test_set.info())
print('\n____________________________________ Split training an test set ____________________________________')
print(len(train_set), "train +", len(test_set), "test examples")
print(train_set.head(4))

# 4.3 Separate labels from data, since we do not process label values
train_set_labels = train_set["GIÁ"].copy()
train_set = train_set.drop(columns = "GIÁ")
test_set_labels = test_set["GIÁ"].copy()
test_set = test_set.drop(columns = "GIÁ")

# 4.4 Define pipelines for processing data.
# INFO: Pipeline is a sequence of transformers (see Geron 2019, page 73). For step-by-step manipulation, see Details_toPipeline.py

# 4.4.1 Define ColumnSelector: a transformer for choosing columns
#chọn cột nào là số nào là chữ để tách riêng ra để xử lý
class ColumnSelector(BaseEstimator, TransformerMixin): #class kế thừa BaseEstimator, TransformerMixin
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):  #tự học dữ liệu
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values

num_feat_names = ['SỐ CHỖ NGỒI', 'NĂM SẢN XUẤT', 'SỐ KM ĐÃ ĐI'] # =list(train_set.select_dtypes(include=[np.number]))
cat_feat_names = ['HÃNG XE', 'KIỂU XE'] # =list(train_set.select_dtypes(exclude=[np.number]))

# 4.4.2 Pipeline for categorical features
# tạo ra pipeline để xử lý dữ liệu 
#xử lý dữ liệu dạng chữ
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)), #chọn ra những giá trị tùm lum như Phường xã ...
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place, xử lý dữ liệu bị khuyết
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors chuyển chữ thành số
    ])

# INFO: Try the code below to understand how a pipeline works
if 10:
    trans_feat_values_1 = cat_pipeline.fit_transform(train_set)

    # The above line of code is equavalent to the following code:
    selector  = ColumnSelector(cat_feat_names)
    temp_feat_values = selector.fit_transform(train_set)
    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)
    temp_feat_values = imputer.fit_transform(temp_feat_values)
    one_hot_encoder = OneHotEncoder()
    trans_feat_values_2 = one_hot_encoder.fit_transform(temp_feat_values)
    if 0:
        # See the encoded features
        print(one_hot_encoder.categories_) # INFO: categories_ is an array of array: categories_[0] is the array of feature 1, categories_[1] is the array of feature 2,...
        # NOTE: OneHotEncoder turns 1 features into N features, where N is the no. of values in that feature
        # e.g., feature "HƯỚNG" having 5 values 'Đông', 'Tây', 'Nam', 'Bắc', 'NO INFO', will become 5 features corresponding with its values
        print(one_hot_encoder.get_feature_names(cat_feat_names))
        print("No. of one-hot columns: " + str(one_hot_encoder.get_feature_names(cat_feat_names).shape[0]))
        print(trans_feat_values_2[[0,1,2],:].toarray()) # toarray() convert sparse to dense array

    # Check if trans_feat_values_1 and trans_feat_values_2 are the same
    #print(trans_feat_values_1.toarray() == trans_feat_values_2.toarray())
    print(np.array_equal(trans_feat_values_1.toarray(), trans_feat_values_2.toarray()))

# 4.4.3 Define MyFeatureAdder: a transformer for adding features "TỔNG SỐ PHÒNG",...
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_TONG_SO_PHONG = True): # MUST NO *args or **kargs
        self.add_TONG_SO_PHONG = add_TONG_SO_PHONG
    def fit(self, feature_values, labels = None):
        return self  # nothing to do here
    def transform(self, feature_values, labels = None):
        SO_PHONG_id, SO_TOILETS_id = 1, 2 # column indices in num_feat_names. can't use column names b/c the transformer SimpleImputer removed them
        # NOTE: a transformer in a pipeline ALWAYS return dataframe.values (ie., NO header and row index)

        TONG_SO_PHONG = feature_values[:, SO_PHONG_id] + feature_values[:, SO_TOILETS_id]
        if self.add_TONG_SO_PHONG:
            eature_values = np.c_[feature_values, TONG_SO_PHONG] #concatenate np arrays
        return feature_values

# 4.4.4 Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), # copy=False: imputation will be done in-place
   # ('attribs_adder', MyFeatureAdder(add_TONG_SO_PHONG = False)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
    ])

# 4.4.5 Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])

# 4.5 Run the pipeline to process training data
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________________________________ Processed feature values ____________________________________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 32 cols of onehotvector for categorical features.' %(len(num_feat_names)))

# (optional) Add header to create dataframe. Just to see. We don't need header to run algorithms
if 0:
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_:
        onehot_cols = onehot_cols + val_list.tolist()
    columns_header = train_set.columns.tolist() + ["TỔNG SỐ PHÒNG"] + onehot_cols
    for name in cat_feat_names:
        columns_header.remove(name)
    processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns = columns_header)
    print('\n____________________________________ Processed dataframe ____________________________________')
    print(processed_train_set.info())
    print(processed_train_set.head())








''' WEEK 04 '''

# In[5]: TRAIN AND EVALUATE MODELS 

# 5.1 Try LinearRegression model
# 5.1.1 Training: learn a linear regression hypothesis using training data 
#import thư viện sklearn 
from sklearn.linear_model import LinearRegression
#tạo 1 cái model class, có nhiều thuộc tính để đưa vào nhưng mình dùng cái ngẫu nhiên của LinearRegression
model = LinearRegression()
#sử dụng hàm fit để học dữ liệu, mình đưa vào feature và labels
model.fit(processed_train_set_val, train_set_labels)
print('\n____________________________________ LinearRegression ____________________________________')
print('Learned parameters: ', model.coef_)
#một bộ số mô tả cho 1 model ở đây ra rất nhiều con số do có nhiều feature hay là 1 parameters sẽ là 1 model thay đổi parameters sẽ ra model khác

# 5.1.2 Compute R2 score and root mean squared error
#hàm có thuật toán tính sai số
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
#R2 score có giá trị càng gần 1 thì càng tốt
#root mean square là giá trị sai số càng nhỏ thì càng tốt 
        
# 5.1.3 Predict labels for some training instances
#lấy một số dữ liệu ra cho nó dự đoán
print("Input data: \n", train_set.iloc[0:9]) 
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
#một số dữ liệu nó lấy ra ngẫu nhiên
#Predictions là giá dự đoán
#Labels là giá chính xác

# 5.1.4 Store models to files, to compare latter
#from sklearn.externals import joblib 
#dùng joblib để lưu model lại
import joblib # new lib
def store_model(model, model_name = ""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    #lệnh dump để down dữ liệu xuống
    joblib.dump(model,'saved_objects/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    #lệnh load để lấy dữ liệu lên
    model = joblib.load('saved_objects/' + model_name + '_model.pkl')
    #print(model)
    return model
store_model(model)


# 5.2 Try DecisionTreeRegressor model
# train bằng DecisionTree
# Training
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ DecisionTreeRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
#R2 score =0.99 số dự đoán rất tốt
#Root Mean Square cũng là số nhỏ
store_model(model)
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


# 5.3 Try RandomForestRegressor model, được đánh giá là 1 trong những model tốt nhất vừa chạy nhanh vừa nhẹ hơn DL
# Training (NOTE: may take time if train_set is large)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5, random_state=42) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ RandomForestRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


# 5.4 Try polinomial regression model
# NOTE: polinomial regression can be treated as (multivariate) linear regression where high-degree features x1^2, x2^2, x1*x2... are seen as new features x3, x4, x5... 
# hence, to do polinomial regression, we add high-degree features to the data, then call linear regression
# 5.5.1 Training. NOTE: may take a while 
from sklearn.preprocessing import PolynomialFeatures
poly_feat_adder = PolynomialFeatures(degree = 2) # add high-degree features to the data
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name = "PolinomialRegression")      
else:
    model = load_model("PolinomialRegression")
# 5.4.2 Compute R2 score and root mean squared error
print('\n____________________________________ Polinomial regression ____________________________________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 5.4.3 Predict labels for some training instances
print("Predictions: ", model.predict(train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


# 5.5 Evaluate with K-fold cross validation 
#dùng cross val score sẽ tự động cắt dữ liệu tự động training và tự động tính gtri trung bình
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
#from sklearn.model_selection import cross_val_predict

#cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
#cv2 = StratifiedKFold(n_splits=10, random_state=42); 
#cv3 = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
print('\n____________________________________ K-fold cross validation ____________________________________')

run_evaluation = 0
if run_evaluation:
    # Evaluate LinearRegression
    model_name = "LinearRegression" 
    model = LinearRegression()              #đưa tên model mà mình muốn
    #cv tương đương với k, sẽ chạy 5 lần
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    #mỗi lần training sẽ được 1 model và nó sẽ tính gtri trung bình
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    #kết quả ra là 5 cái sai số của 5 lần chạy, lần thứ 3 là ra kết quả tốt nhất

    # Evaluate DecisionTreeRegressor
    model_name = "DecisionTreeRegressor" 
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    #kết quả còn tệ hơn cái model LinearRegression nữa, lần trước ra kết quả tốt 
    # là do chạy trên tập dữ liệu đã được train, đó là hiện tượng overfiting

    # Evaluate RandomForestRegressor
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    #kq tốt hơn hẳn cái ở trên nên gtri trung bình sẽ tốt hơn

    # Evaluate Polinomial regression
    model_name = "PolinomialRegression" 
    model = LinearRegression()
    nmse_scores = cross_val_score(model, train_set_poly_added, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    #kq ra số lớn không tốt
    #tốt nhất là RandomForestRegressor
else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "PolinomialRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')


# In[6]: FINE-TUNE MODELS 
# NOTE: this takes TIME
# INFO: find best hyperparams (param set before learning, e.g., degree of polynomial in poly reg, no. of trees in rand forest, no. of layers in neural net)
# Here we fine-tune RandomForestRegressor and PolinomialRegression
print('\n____________________________________ Fine-tune models ____________________________________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    print('Best estimator: ', grid_search.best_estimator_)  
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

method = 2
# 6.1 Method 1: Grid search (try all combinations of hyperparams in param_grid)
if method == 1:
    from sklearn.model_selection import GridSearchCV
    
    run_new_search = 0      
    if run_new_search:
        # 6.1.1 Fine-tune RandomForestRegressor
        #vì randomforest là model tốt nhất nên mình sẽ finetune nó
        model = RandomForestRegressor(random_state=42)
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]} ]
            # Train across 5 folds, hence a total of (12+6)*5=90 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor") 
        #tổng cộng có 90 lần chạy 18 bộ số mỗi cái có 1 thông số
        # nên dùng rmse = 16979874072.6 {'bootstrap': True, 'max_features': 8, 'n_estimators': 3}  

        # 6.1.2 Fine-tune Polinomial regression     
        # thử polinomial vì có thể bậc 3 4 5 sẽ tốt hơn     
        model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()), # add high-degree features
                           ('lin_reg', LinearRegression()) ]) 
        param_grid = [
            # try 3 values of degree
            {'poly_feat_adder__degree': [1, 2, 3]} ] # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
            # Train across 5 folds, hence a total of 3*5=15 rounds of training 
            #thử discerd 5 lần ra 15 vòng lặp
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/PolinomialRegression_gridsearch.pkl') 
        print_search_result(grid_search, model_name = "PolinomialRegression") 
        #kết quả ra là nên dùng bậc 1
    else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        grid_search = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name = "PolinomialRegression") 

#model tốt nhất là random forest


# In[7]: ANALYZE AND TEST YOUR SOLUTION
# NOTE: solution is the best model from the previous steps. 

# 7.1 Pick the best model - the SOLUTION
# Pick Random forest
#load lên
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_
# Pick Linear regression
#best_model = joblib.load('saved_objects/LinearRegression_model.pkl')

#chạy solution của best model
print('\n____________________________________ ANALYZE AND TEST YOUR SOLUTION ____________________________________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUION")   

#random forest cho cái nhìn sâu vào trong dữ liệu mà những model khác không có
# 7.2 Analyse the SOLUTION to get more insights about the data
# NOTE: ONLY for rand forest
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    #feature_importances là feature nào quan trọng (đứng đầu danh sách)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + ["TỔNG SỐ PHÒNG"] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

# 7.3 Run on test data
processed_test_set = full_pipeline.transform(test_set)  
# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("Predictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')


'''
Kết quả hiện chưa tốt. 
Gợi ý cả thiện:
1. Xóa các samples bất thường (giá quá cao, quá thấp)
2. Xóa cột HƯỚNG, hoặc tốt nhất là thêm dữ liệu (2000+)
'''


# In[8]: LAUNCH, MONITOR, AND MAINTAIN YOUR SYSTEM
# Go to slide: see notes

done = 1
