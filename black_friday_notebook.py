# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import pearsonr

# some imports
#from sklearn_pandas import DataFrameMapper
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, ShuffleSplit
from sklearn.decomposition import PCA, NMF 
    # Non-Negative Matrix Factorization (NMF)
    # Find two non-negative matrices (W, H) whose product approximates the non- negative matrix X. 
    # This factorization can be used for example for dimensionality reduction, source separation or topic extraction.
#from sklearn.svm import SVR
from sklearn import linear_model, neighbors, ensemble
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
#from sklearn.kernel_ridge import KernelRidge
import xgboost

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn import model_selection, feature_selection, metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import f_regression, SelectKBest # feature extraction with Statistical Selection

from xgboost.sklearn import XGBRegressor
import matplotlib.pylab as pyl
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 8
plt.style.use("ggplot")
sns.set_style("white")
#%%
train = pd.read_csv("K:/datasets/black friday/data/train.csv")
test = pd.read_csv("K:/datasets/black friday/data/test.csv")

#train = pd.read_csv("E:/datasets/black friday/data/train.csv")
#test = pd.read_csv("E:/datasets/black friday/data/test.csv")


data1 = train.copy(deep=True)
data_val = test.copy(deep=True)

data_cleaner = [data1, data_val]

target = "Purchase"

# we are provided with a sample submission that consists of values in 3 columns:
# User_ID, Product_ID and Purchase. The test data is missing the Purchase column. 
# User_ID and product categories are included in the test data.
# We are asked to predict the Purchase amount for a given product 
# and a known customer based on historic data for that same customer and purchase history - 
# but also for products first bought by the same customer!
#%%
train.head(20)
#%%
print(train.info())
print(test.info())
# NaN only in Product Category 2 and 3
#%%
train.sample(5)
#%%
test.head(20)
#%%
train.describe()
#%%
## Some thoughts
# Purchase variable is our outcome or dependent variable. Its continuous values are int64 datatype.

## User_ID, Gender, Product_ID, City_Category, Occupation, Marital_Status, Product_Category_1, Product_Category_2, Product_Category_3 are nominal datatypes.
## Stay_In_Current_City_Years is an ordinal datatype.
## we will take a closer look at what we can already use from those and do some conversions
## LabelEncoding and dummy variables come to mind

## Age already comes as a binned / interval datatype and is not continuous anymore. We may have to LabelEncode this.

## Occupation is an already LabelEncoded nominal datatype, we can work with this and maybe get dummies 

## Marital_Status comes as a binary nominal datatype, get dummies from that

## Product_Category_1 through _3 come as already LabelEncoded nominal datatypes. However we have many missing values in 2 and 3. 
## Missing values should probably not be engineered but assigned a unique "missing" category
#%%
def identical_categories(datatrain, datatest):
    df = pd.DataFrame([], datatest.columns)
    df_dict = {"items_only_in_train": 0, "items_new_in_test": 0}
    df = df.assign(**df_dict)
    for col in datatest.columns:
        if (pd.DataFrame(datatrain.sort_values(col)[col].unique()).count() == pd.DataFrame(datatest.sort_values(col)[col].unique()).count())[0]==False:
            items_only_in_train = len(set(datatrain[col]) - set(datatest[col]))
            new_in_test = len(set(datatest[col]) - set(datatrain[col]))
            df.loc[col, "items_only_in_train"] = items_only_in_train
            df.loc[col, "items_new_in_test"] = new_in_test
    return df

identical_categories(train,test)
# we can see that we have new values in test only for Product_ID. 
# This is noise and we will set it to -1.
#%%
# store the indices for labelencoding later (set unknown ids in test to -1)
product_ids_train_only = (set(train["Product_ID"]) - set(test["Product_ID"])) # 186 are absent in test
new_product_ids_test = (set(test["Product_ID"]) - set(train["Product_ID"]) ) # 46 are new in test
#%%
"""
# =============================================================================
# Some EDA
# =============================================================================

#%%
fig, saxis = plt.subplots(2,3, figsize = (16,12))
sns.barplot(x = "Gender", y="Purchase", hue = "Marital_Status", data = train, ax= saxis[0,0])
sns.barplot(x = "Age", y="Purchase", hue = "City_Category", data = train, ax= saxis[0,1])
sns.barplot(x = "Occupation", y="Purchase", hue = "Stay_In_Current_City_Years", data = train, ax= saxis[0,2])

sns.barplot(x = "City_Category", y="Purchase", data = train, ax= saxis[1,0])
sns.barplot(x = "Marital_Status", y="Purchase", data = train, ax= saxis[1,1])
sns.barplot(x = "Stay_In_Current_City_Years", y="Purchase", data = train, ax= saxis[1,2])
# interesting enough, Marital Status has no apparent effect on Purchase. Stay in the current city also seems to be a weak predictor,
# since it goes up and down alternatively with the years spent in one city. 
# If we look at it like a categorical variable we can see that 2 years yields the largest amount for Purchase.
# City Category shows some promise. Surprisingly to me, Age has only an effect for some city categories and 
# I can't recognize it as being consistent.
# Gender makes a difference. Men as a group apparently spend more. Are those men married? Let's take a closer look at that later.
# We need to take a closer look at Occupation later to discern anything useful from that variable
#%%
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (16,12))
sns.boxplot(x = "Gender", y= "Purchase", hue = "Marital_Status", data = train, ax=ax1)
sns.boxplot(x = "Gender", y= "Purchase", hue = "City_Category", data = train, ax=ax2)
sns.boxplot(x = "Gender", y= "Purchase", hue = "Stay_In_Current_City_Years", data = train, ax=ax3)

# interesting! Here we can see that married men buy _less_ and women have roughly the 
# same purchase amount regardless of being married or unmarried. So apparently in this case the bias that married 
# men have to spend more for their spouses is untrue, at least for purchases from this company. 
# Or maybe they can buy less here because they have to spend more money on other things once they are married? We'll never know :) 
# At least the variable Marital Status seems to be a viable predictor for men, although the relative difference in amount is small.
# We can also see something more significant: men spend more than women in total amount!

# Men and women from City Category C spend more compared to the other categories. Men from that category 
# spend significantly more than women from that category and men in general spend more than women, 
# as we have seen in the first graph on the left.
# Stay in the current city is almost uneventful for women. 
# Only when they first arrive will they spend less than in the following years. 
# For men, it's up and down with two years resulting in most spending.
#%%
plt.figure(figsize=[16,12])
plt.subplot(231)
plt.hist(x= [train[train["Age"] < "26-35"]["Purchase"], train[train["Age"] >= "26-35"]["Purchase"]], stacked = True, label = ["Age 26-35", "older 36"], bins=30)
plt.title("Purchase Histogram by Age")
plt.xlabel("Purchase amount in $")
plt.ylabel("# of customers")
plt.legend()

plt.subplot(232)
plt.hist(x= [train[train["Gender"] == "F"]["Purchase"], train[train["Gender"]== "M"]["Purchase"]], stacked = True, label = ["Female", "Male"], bins = 30)
plt.title("Purchase Histogram by Gender")
plt.xlabel("Purchase amount in $")
plt.ylabel("# of customers")
plt.legend()
#%%
train["Gender"].value_counts()
# we can see the reason why men seem to spend more than women in relative terms: 
# significantly more men buy from that company over every age group and purchase amount. It seems natural that men should spend much more than
# women in absolute purchase amount.
# Most of the customers make purchases for amounts around 7000 $, some customers above 35 years of age have purchases for the largest amounts.

# female to male ratio of buyers is
fm_fac = 135809/414259
#%%
print("relative purchase amount females /  males: ")
print(round(train[train["Gender"] == "F"]["Purchase"].sum()/(train[train["Gender"]== "M"]["Purchase"]*fm_fac).sum(),3))
# We can see that females make purchases for only slightly less value than males relative to their representation in the train data

#%%
# Category variable values
l = [data1["Product_Category_1"].value_counts(),
data1["Product_Category_2"].value_counts(),
data1["Product_Category_3"].value_counts()]
for i in l:
    print(i)

#%%
print(data1[["Product_ID", target]].groupby("Product_ID", as_index = False).max().sort_values(target, ascending = False))
print("-"*10)
data1["Product_ID"].value_counts()
#%%
sns.barplot(x = "Product_ID", y="Purchase", data = train)
#%%
# the next parts are just interesting to me but in all likelihood not relevant to the modeling phase

plt.figure(figsize=[24,16])
plt.subplot(231)
plt.hist(x= data1["Product_ID"].value_counts(), bins=65)
plt.title("Product_ID Histogram")
plt.xlabel("Product_ID")
plt.ylabel("# of purchases for that ID")
## We can see that we have many more products that are being sold less individually
## and less products that are sold very often individually.
## Lets look at the relationship of the variables Purchase and Product_ID a little closer
#%%
## Here we look at the total turnover of each product
print(data1[["Product_ID", target]].groupby("Product_ID", as_index = False).sum().sort_values(target, ascending = False))

## interesting, in some cases we have quite a big turnover per product. What is the total turnover?
print("-"*10)
print("total turnover is: \n", data1[["Product_ID", target]].groupby("Product_ID", as_index = False).sum()["Purchase"].sum()/1000000, " million $")
## so we have quite a big company with over 5 billion $ turnover - assuming that the value of Purchase is given in USD. 
#%%
# our purchase sum for any unique product ID
data1[["Product_ID", target]].groupby("Product_ID", as_index = False).sum()["Purchase"]
#%%
# more visualization on products and purchases
turnover = data1[["Product_ID", target]].groupby("Product_ID", as_index = False).sum()["Purchase"]
## Let's visualize this a bit better
plt.figure(figsize=(24,16))

plt.subplot(231)
plt.hist(x=turnover, bins=55)
plt.title("Purchase Histogram")
plt.xlabel("Purchase sum per product ID [$]")
plt.ylabel("# of purchases")

# Data are very skewed. Let's apply log10

#apply log to data for interpretability
plt.subplot(232)
plt.hist(x=np.log10(turnover), bins=55)
plt.title("Purchase Histogram log")
plt.xlabel("Log Purchase sum per product ID [$]")
plt.ylabel("# of log purchases")
plt.show()

# it looks a little more like a normal distribution. What's the value in that for us here? Would love advice or insights.
#%%
#our counts of the purchases for any unique product ID
data1[["Product_ID", target]].groupby("Product_ID", as_index = False).count()["Purchase"]

#%%
# more visualization on products and purchases - count of purchases for unique product ID
products_num_purchases = data1[["Product_ID", target]].groupby("Product_ID", as_index = False).count()["Purchase"]
plt.figure(figsize=(24,16))

plt.subplot(231)
plt.hist(x= products_num_purchases, bins=55, stacked= True, label =["products"])
plt.title("Product Purchase Count Histogram")
plt.xlabel("# of purchases per product ID")
plt.ylabel("# of products")
plt.legend()
           
# Data are very skewed. Let's apply log10

#apply log to data for interpretability
plt.subplot(232)
plt.hist(x=np.log10(products_num_purchases), bins=55, stacked=True, label =["products"])
plt.title("Product Purchase Count Histogram")
plt.xlabel("log # of purchases per product ID")
plt.ylabel("# of log purchases per product ID")
plt.legend()
plt.show()

# in what way does log transformation help us with interpretability?
# to my eye, it looks like the untransformed graph gives us more 
## (easily accessible?) information
# does this make sense for categorical (as opposed to continuous) information?
# Advice & comments welcome!

## We can conclude that most purchases have a small turnover and some purchases have a large turnover
## Also, most products have small amount of purchases, some products have a large amount of purchases
#%%
# how many unique products do we have in train (we know it's more than in test)? Double Check
data1[["Product_ID", target]].groupby("Product_ID", as_index = False).sum().sort_values(target, ascending = False)["Product_ID"].nunique()
#%%
## what is the distribution of our purchase amount vs # of purchases for that amount
plt.figure(figsize=(12,8))
plt.hist(x= (data1["Purchase"]), bins=45)
plt.title("Purchase Histogram")
plt.xlabel("Purchase amount in $")
plt.ylabel("# of purchases for that amount")
plt.show()
##
#%%
# How many customers did make more than one purchase? Counts of Purchases for customer
cust = data1[["User_ID", target]].groupby("User_ID", as_index = False).count()["Purchase"]

plt.figure(figsize=(12,8))
plt.hist(x= cust, bins=45)
plt.title("Purchase per customer Histogram")
plt.xlabel("Purchase count")
plt.ylabel("# of customers with that count of purchases")
plt.show()
# similar distribution as for the Product_ID count.

"""

#%%
# we label encode the different variables
# label encoding for different algorithms
label = LabelEncoder()  #encodes objects to categorical integers 
columns_to_encode = ["User_ID", "Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]
for col in columns_to_encode:
    data_cleaner[0][col+"_code"] = label.fit_transform(data_cleaner[0][col]) 
    data_cleaner[1][col+"_code"] = label.transform(data_cleaner[1][col])  

#%%
# do the product id codes manually, since we have different codes in train and test
# since the test set contains previously unseen labels by the LabelEncoder we get an error and have to change those manually to and arbitrary value like -1

# could label encode the product ids that are found in train with labels from train 
label = LabelEncoder()  
label.fit(data1["Product_ID"])

data1["Product_ID_code"] = label.transform(data1["Product_ID"]) 
data_val["Product_ID_code"] = 0
data_val.loc[~data_val["Product_ID"].isin(new_product_ids_test), "Product_ID_code"] = label.transform(data_val.loc[~data_val["Product_ID"].isin(new_product_ids_test), "Product_ID"]) 
#%%
# could set the others (new ones and only in test!) to -1, for example
data_val.loc[data_val["Product_ID"].isin(new_product_ids_test) , "Product_ID_code"] = -1 
# the correct label encoding in test gave us over 200 RMSE decrease.
#Removing noise is important!
#%%

# filling NaN values with 0. 0 was not present in categories before the fill
# change dtype to int, since we only have .0 values in there
for dataset in data_cleaner:
    
    dataset["Product_Category_2"].fillna(0, inplace = True) # compare RMSE without category 2 and 3 later and without imputing!
    dataset["Product_Category_3"].fillna(0, inplace = True)
    dataset["Product_Category_2"] = pd.DataFrame(dataset["Product_Category_2"], dtype = "int64")
    dataset["Product_Category_3"]  = pd.DataFrame(dataset["Product_Category_3"], dtype = "int64")

#%%
# for different tests
data1_code_purchase = ['Occupation', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3', 'User_ID_code',
       'Product_ID_code', 'Gender_code', 'Age_code', 'City_Category_code',
       'Stay_In_Current_City_Years_code', 'Purchase']

data1_code = ['Occupation', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3', 'User_ID_code',
       'Product_ID_code', 'Gender_code', 'Age_code', 'City_Category_code',
       'Stay_In_Current_City_Years_code']

# target is the Purchase column
#%%
## Can we glean any insights from a pairplot? 
#sns.pairplot(data1[data1_code_purchase])
## Too many datapoints, now new insights for me. Log transform helpful here? Not on objects though
#%%
def correlation_heatmap(df):
    hm , ax = plt.subplots(figsize=(12,14))
    colormap = sns.diverging_palette(220,10, as_cmap = True)
        
    hm = sns.heatmap(
            data=df.corr(), # e.g. "pearson" or "spearman" 
            cmap=colormap,
            square = True,
            cbar_kws = {"shrink": .9},
            ax=ax,
            annot=True,
            linewidths = 0.1,
            vmax=1.0,
            linecolor="white",
            annot_kws={"fontsize": 12}
            )

    plt.title("Pearson Corr of Features", y=1.05, size=15)
#%%
correlation_heatmap(data1[data1_code_purchase])
# we see some weak correlations with the target, product categories 1 and 2 and product ID have a larger correlation
#%%
"""
correlations = {}
features = data1_code # without purchase
for f in features:
    data_temp = data1[[f, target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0] # could also use spearman or sth else?
data_correlations = pd.DataFrame(correlations, index = ["Value"]).T #T transposes the column names to index and the values to the Value-ex-index
data_correlations.loc[data_correlations["Value"].abs().sort_values(ascending = False).index] #gets the original values sorted by abs()

# we see weak linear correlations. How about non linear correlations?
# could try this in the future with another function
"""
#%%
# We saw in the initial analysis that test and train data are fairly similar 
# (User and Product IDs are almost identical)
# We marked the instances in test, which are not present in train, with -1

#%%
data1.info() # all the columns have the datatypes we want
#%%
# =============================================================================
# # create interaction terms for regression
# # we took a long time to evaluate this but ultimately we 
# # decided to go with the original dataset without interaction terms
# # based on RMSE CV results
# =============================================================================
"""
# pick the columns we want to create interaction terms 
#for regression
columns_interaction = [col for col in data1.columns if col in data1_code_purchase]

scaled_data = data1[columns_interaction] # drop all the non encoded columns in the new df
#%%
scaled_data.info()
#%%
# =============================================================================
# need to scale data before creating interaction terms
# =============================================================================
predictors = scaled_data.drop(target,axis=1).columns

scaler = MinMaxScaler(feature_range=(1e-4, 1)) # try out effect of minmax instead of standardscaler for feature creation
# use feature range without 0 to avoid creating inf or nan values when creating interaction terms
scaled = scaler.fit(scaled_data[predictors])
scaled_data = scaled.transform(scaled_data[predictors])
scaled_data = pd.DataFrame(scaled_data, columns = predictors )
scaled_data = pd.concat([scaled_data, data1[target]], axis=1)
scaled_data.head()
#%%
# function to create interaction variables

def create_interaction_variables(data, columns):
    numerics = data.loc[:, columns] # apply this only on numeric columns without target column
    # for each pair of variables, determine which mathmatical operators to use
    for i in range(0, numerics.columns.size-1):
        for j in range(0, numerics.columns.size-1):
            col1 = str(numerics.columns.values[i])
            col2 = str(numerics.columns.values[j])
            # multiply fields together (we allow values to be squared)
            if i <= j:
                name = col1 + "*" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name = name)], axis = 1)
            # add fields together
            if i < j:
                name = col1 + "+" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis = 1)
            # divide and subtract fields from each other
            if not i == j:
                name = col1 + "/" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name = name)], axis = 1)
                name = col1 + "-" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name= name)], axis = 1)
        print("Column {} done, moving on to next column.".format(col1))
    return data

predictors = scaled_data.drop(target,axis=1).columns
scaled_data = create_interaction_variables(scaled_data, predictors)
#%%
pd.set_option("display.max_columns",13)
scaled_data.info()
#292 columns, so created roughly 280
#%%
scaled_data.sample(10)
#%%
# opt to drop the columns with nan or inf values
def del_nan_or_inf_cols(data):
    for col in data.columns:
        if data[col].dtype == np.dtype('O'): #needed that for another approach. 
                                            #We already removed the object columns
            data.drop([col], axis=1, inplace=True)
            print("dropped col because it's object type: ",col)
            continue
        if np.all(~np.isnan(data[col])) and np.all(np.isfinite(data[col])):
            continue
        else:
            data.drop([col], axis=1, inplace=True)
            print("dropped col because it has nan or inf values in it: ",col)
#    return data

del_nan_or_inf_cols(scaled_data)
#%%
True in scaled_data.isnull()
# no null values in the df! - thats why our function didn't detect any inf or NaN
# we checked for infinite values with a for loop and read through it by hand
#%%
# function to remove correlated variables
def remove_correlated_variables(data, target, inplace = False):
    # calculate the correlation matrix for the predictors only
    df_corr = data.corr(method='spearman')
    # create a mask to ignore self-correlation
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
    
    drops = []
    # loop through each variable, drop the target from analysis though
    for col in df_corr.drop(target,axis=1).columns.values:
        # if we've already determined to drop the current variable, continue
        if np.isin([col], drops):
            continue
        # find all the variables that are highly correlated with the current variable
        # and add them to the drop list
        corr = df_corr[abs(df_corr[col]) > 0.9].index #remove all above 0.9x correlation-we varied x for tests
        drops = np.union1d(drops, corr)
    
    print("\nDropping", drops.shape[0], "highly correlated features...n", drops)
    return data.drop(drops, axis=1, inplace=inplace)

remove_correlated_variables(scaled_data, target, inplace = True)
#%%
#VIF Formula:
r = 0.9 #(result is vif of ~5)
#r = 0.95 #(result is vif of ~10)
vif = 1/(1-r**2)
vif
""" 
#%%
# =============================================================================
# # we can also let XGB do the feature selection with feature importances! - see below
# # remove correlated variables
# =============================================================================
"""
# =============================================================================
# adding original variables and target to the dataset
# =============================================================================

data = scaled_data
predictors = data.drop(target, axis=1).columns.values
# add all the original predictors to the dataset and let XGB find the best predictor set for best CV score
# by using the feature importances of XGB
for x in data1[data1_code].columns:
    if x not in predictors:
        data[x] = data1[x]

# updating predictors
predictors = data.drop(target, axis=1).columns.values
data.info()
"""
#%%

# =============================================================================
# Test CV data with KFold
# Splitting strategy
# =============================================================================

target = "Purchase"

# for time reasons we decided on 5 fold cv
cv_split = KFold(n_splits = 5, shuffle = True, random_state = 42)


#%%
"""
params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'nthread' : 8,
    # Other parameters
    'objective':'reg:linear',
    #'booster':'gblinear', # instead of gbtree for testing?
    'seed' : 42,
}
"""
#%%

"""
# =============================================================================
# # need some feature selection because we run into memory issues. Maybe try out PCA as well?
#  cross validate SKBest with xgboost.train  #########
# =============================================================================


#data = data1

data = scaled_data # first run without removing VIF columns
#data = scaled_data_vif

#data = pd.concat([scaled_data.loc[:, sorted_idx[::-1].index.values], scaled_data[target]], axis=1)
predictors = data.drop(target, axis=1).columns.values
#predictors = data1_code


CVCompare_columns = ["selected k", "RMSE train", "RMSE test", "RMSE test 3*STD", "boost_rounds", "time"]
CVCompare = pd.DataFrame(columns=CVCompare_columns)
row_index = 0


num_boost_round = 100000
early_stopping_rounds=100

# create lists to store train and validation CV scores after each full kfold step with all iterations
train_score_CV = []
val_score_CV = []
#create lists to store std scores for every iteration (all folds)
train_acc_std = []
val_acc_std = []

for k in [10,"all"]: #values for select k best
    #set up lists for results
    train_score = []
    val_score = []
    boost_rounds = []
    start = time.perf_counter()
    for n, (train_i, test_i) in enumerate(cv_split.split(data[predictors], data[target])): 
        # use kfold and "average" over the whole dataset, use early stopping in xgboost.train for every eval_set
        print("\nFitting CV folds k = {}...".format(n+1))    
        X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
        y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]
                
        print("CV with selected {} best features now".format(k))
        
        # select k best features - univariate selection
        kbest = SelectKBest(k=k, score_func = f_regression)
        kbest_picker = kbest.fit(X_train,y_train)
        kbest_picked_train = kbest_picker.transform(X_train)
        kbest_picked_test = kbest_picker.transform(X_val) 
        columns_i = kbest_picker.get_support(indices=True)
        labels = []
        for i in columns_i:
            labels.append(predictors[i])
        print("\nLabels for this fold are: ",labels)  #or: labels = predictors[kbest_picker.get_support()]
        print("Scores are: {}".format(kbest_picker.scores_[kbest_picker.get_support()]))
        #DMatrix for every train and val set in folds
        dtrain = xgboost.DMatrix(kbest_picked_train, label=y_train.values, feature_names = labels ,nthread = 8)
        dtest = xgboost.DMatrix(kbest_picked_test, label=y_val.values, feature_names = labels, nthread = 8)

        # fit the model only using that many training examples # test if using evals set and best iterations is ok
        reg = xgboost.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtest, "Test")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval = 300
                    )
        
        #print and store boost rounds
        print("Best RMSE: {:.2f} in {} rounds".format(reg.best_score, reg.best_iteration+1))  
        boost_rounds.append(reg.best_iteration)
        #calculate the training accuracy 
        trainpreds = reg.predict(dtrain,  ntree_limit=reg.best_ntree_limit)
        train_accuracy = np.sqrt(mean_squared_error(
	                                   y_train,
	                                    trainpreds)) 
        #calculate the validation accuracy 
        valpreds =  reg.predict(dtest, ntree_limit=reg.best_ntree_limit)
        val_accuracy = np.sqrt(mean_squared_error(
	                                    y_val,
	                                    valpreds
	                                    ))
   
        # store the scores in their respective lists
        train_score.append(train_accuracy)
        val_score.append(val_accuracy)
        
    #store mean boost rounds over folds
    boost_rounds = np.mean(boost_rounds)
    # store the stds of the cv fold values
    train_acc_std = (np.std(train_score))
    val_acc_std = (np.std(val_score))
    # store means of the individual scores to get final cv scores
    train_score_CV = (np.mean(train_score))
    val_score_CV = (np.mean(val_score))
    
    # store results in DF
    CVCompare.loc[row_index, "selected k"] = k
    CVCompare.loc[row_index, "RMSE train"] = train_score_CV
    CVCompare.loc[row_index, "RMSE test"] = val_score_CV
    CVCompare.loc[row_index, "RMSE test 3*STD"] = val_acc_std*3
    CVCompare.loc[row_index, "boost_rounds"] = boost_rounds
    duration = time.perf_counter() - start
    CVCompare.loc[row_index, "time"] = duration
    row_index+=1

# print and sort table:
CVCompare.sort_values(by= ["RMSE test"], ascending = True, inplace=True)
print(CVCompare)
#print("done with columns: ", predictors)
#

# data1 dataset:
#  selected k RMSE train RMSE test RMSE test 3*STD boost_rounds     time (eta 0.3)
#1         10    2227.22   2489.94         25.6293       1846.8  1846.37
#2        all    2233.16   2490.33         25.1451         1780  2054.27
#0          8    2536.08    2634.5         31.6253        869.4  864.156

# with untuned XGB its questionable to assess if we can drop one predictor (10 instead of 11) out of our set.
# might have to increase cv folds to 10

# compare results above with interaction predictors
#  selected k RMSE train RMSE test RMSE test 3*STD boost_rounds     time (eta 0.3)
#2         20    2464.84   2635.72         30.3582        981.6  489.783
#1         10     2576.8   2667.27         24.6173        721.8  265.274
#0          5    2615.93   2676.76         27.9883          703   194.07

### so for XGB we could do a different selection process making use of the 
feature importance scores instead of the f-statistic with select k best
# by selecting the most important features with an importance bigger than 5% 
of the most important features importance during splits... see approach in cells below

# first though we try out applying PCA below - 
# PCA delivered bad results. Not using PCA on that dataset

# using VIF/correlation removal on features results in 19 predictors left and we test this now with select k best:
#  selected k RMSE train RMSE test RMSE test 3*STD boost_rounds     time (eta 0.3)
#3         13    2197.51   2512.64         20.5489       1838.2  646.812
#1         11    2179.09    2513.1         20.4454       2003.4  663.265
#2         12    2193.16   2513.11         20.4272         1878   655.95
#0         10    2189.34   2513.32         18.3596       1967.6  630.218
#6         16    2149.79   2520.04         19.6398         1933  778.095
#5         15    2166.81   2520.35          21.767       1875.4   727.99
#4         14    2179.57   2521.23         27.9491       1860.2  679.577
"""
#%%
"""
# =============================================================================
#  cross validate SKBest with xgboost.cv  
# # =============================================================================

params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'nthread' : 8,
    # Other parameters
    'objective':'reg:linear',
    #'booster':'gblinear', # instead of gbtree for testing?
    'seed' : 42,
}


#predictors = data1_code
#data = data1.iloc[0:1000,:]
data = scaled_data#.iloc[0:20000,:] # first run without removing VIF columns
#data = scaled_data_vif

#data = pd.concat([scaled_data.loc[:, sorted_idx[::-1].index.values], scaled_data[target]], axis=1)
predictors = data.drop(target, axis=1).columns.values

CVCompare_columns = ["selected k", "RMSE train", "RMSE test", "RMSE test 3*STD", "boost_rounds", "time"]
CVCompare = pd.DataFrame(columns=CVCompare_columns)
row_index = 0


seed = 42
num_boost_round = 1000000
early_stopping_rounds=100
metrics = {'rmse'}
verbose_eval = 100
nfold = 5
folds = cv_split

#values for select k best
for k in [150,100,9,10,11,12,13,15,20,25,30,35,37,40,45,50,80]: # "all" is too much for my memory
    #set timer
    start = time.perf_counter()
    
    # select k best features - univariate selection
    kbest = SelectKBest(k=k, score_func = f_regression)
    kbest_picker = kbest.fit(data[predictors],data[target])
    kbest_picked_train = kbest_picker.transform(data[predictors])
    labels = np.array(predictors)[kbest_picker.get_support()]
    print("\nLabels for this fold are: ",labels)
    print("Scores are: {}".format(kbest_picker.scores_[kbest_picker.get_support()]))

    #create DMatrix for whole train set
    dtrain = xgboost.DMatrix(kbest_picked_train, label=data[target].values, feature_names = labels ,nthread = 8)
    
    # Run CV
    cv_results = xgboost.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        seed=seed,
        nfold=nfold,
        folds = folds,
        metrics=metrics,
        verbose_eval= verbose_eval,
        early_stopping_rounds=early_stopping_rounds
                )
        
    # get best RMSE
    results_train = cv_results['train-rmse-mean'].min()
    results_test = cv_results['test-rmse-mean'].min()
    results_test_std = cv_results['test-rmse-std'][cv_results['test-rmse-mean'].idxmin()]
    boost_rounds = cv_results['test-rmse-mean'].idxmin() + 1 
    
    # store results in DF
    CVCompare.loc[row_index, "selected k"] = k
    CVCompare.loc[row_index, "RMSE train"] = results_train
    CVCompare.loc[row_index, "RMSE test"] = results_test
    CVCompare.loc[row_index, "RMSE test 3*STD"] = results_test_std*3
    CVCompare.loc[row_index, "boost_rounds"] = boost_rounds
    duration = time.perf_counter() - start
    print("\tRuntime of all CV folds for {} features was {:.0f}:{:.0f}:{:.1f}".format(k,
          duration // 3600, (duration % 3600 // 60), duration % 60))
    CVCompare.loc[row_index, "time"] = "{:.0f}:{:.0f}:{:.1f}".format(\
      duration // 3600, (duration % 3600 // 60), duration % 60)
    row_index+=1

# print and sort table:
CVCompare.sort_values(by= ["RMSE test"], ascending = True, inplace=True)
print(CVCompare)
#"""
# results for skbest values
"""
   selected k RMSE train RMSE test RMSE test 3*STD boost_rounds       time
0         150    2186.47   2525.33          110.19         1328  0:53:41.8
1         100    2144.73   2545.77         116.267         1754  0:50:17.7
7          15    2566.67   2786.75         127.962         1363   0:7:56.0
6          13    2561.31   2788.04         129.018         1415   0:7:53.5
4          11    2567.83   2788.42         132.498         1387   0:7:10.6
5          12    2578.93   2788.45         134.128         1292   0:7:11.9
3          10     2541.7   2788.74         129.702         1656   0:8:24.9
2           9     2541.9   2788.82         129.876         1654   0:7:59.2
8          20    2542.48   2789.77         129.045         1473  0:10:47.0
10         30    2554.09   2794.45         123.083         1157  0:11:31.3
9          25    2577.12   2796.84         128.018         1104   0:9:51.3
14         45    2586.35    2799.2         123.471          885  0:13:17.9
11         35    2556.33    2799.2         126.085         1076  0:12:22.5
12         37    2548.66   2799.77         120.098         1125  0:13:32.7
13         40    2544.54   2800.16         125.551         1143  0:14:34.4
15         50    2564.58   2801.13         120.413          997  0:15:33.4
16         80    2533.08   2804.67         127.259         1064  0:24:40.4
"""
# should have gone with removing correlated columns first and then checking 
# if we can find a better combination with less RMSE than the original dataset
# but we tried both ways in the end and we could not improve RMSE unfortunately.
#%%
"""
# =============================================================================
# # get feature importances for XGB and test effect of selecting 
# # only a subset (the most important features over a certain 
# # threshold of importance)
# # selection by CV score 
# =============================================================================

#%%
# =============================================================================
# # CV with feature importances from of XGB 
# =============================================================================
#data = data1.iloc[0:1000,:]
data = scaled_data#.iloc[0:20000,:] # reduce dataset size to check the code

predictors = data.drop(target, axis=1).columns.values
#predictors = data1_code
# reference the feature list for later use in the feature importance section 

# create lists to store train and validation CV scores after each full kfold step with all iterations
train_score_CV = []
val_score_CV = []
#create lists to store std scores for every iteration (all folds)
train_acc_std = []
val_acc_std = []

num_boost_round = 100000
early_stopping_rounds=100

X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=0.1, shuffle = True, random_state=42)

#DMatrix for every train and val set in folds
dtrain = xgboost.DMatrix(X_train, label=y_train.values, feature_names = predictors, nthread = 8)
dtest = xgboost.DMatrix(X_test, label=y_test.values, feature_names = predictors, nthread = 8)

# fit the model ####  test if using evals set and best iterations is ok
reg = xgboost.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, "Test")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval = 300
            )
 
#print and store boost rounds
print("Best RMSE: {:.2f} in {} rounds".format(reg.best_score, reg.best_iteration+1))  

#calculate the training accuracy 
trainpreds = reg.predict(dtrain,  ntree_limit=reg.best_ntree_limit)
train_accuracy_base = np.sqrt(mean_squared_error(
	                                   y_train,
	                                    trainpreds)) 

#calculate the validation accuracy 
valpreds =  reg.predict(dtest, ntree_limit=reg.best_ntree_limit)
val_accuracy_base = np.sqrt(mean_squared_error(
	                                    y_test,
	                                    valpreds
	                                    ))
feature_importance = pd.Series(reg.get_score(importance_type='weight')).sort_values(ascending=False)
  
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

#Best RMSE: 2514.21 in 1451 rounds
#%%
#k: A threshold below which to drop features from the final data set. 
# the percentage of the most important feature's importance value
# Can cycle through threshold with CV

seed = 42
num_boost_round = 100000
early_stopping_rounds=100
metrics = {'rmse'}
verbose_eval = 300
nfold = 5
folds = cv_split

CVCompare_columns = ["threshold k", "RMSE base train", "RMSE base test", "CV RMSE train", "CV RMSE test", "CV RMSE test 3*STD", "CV boost_rounds", "time"]
CVCompare = pd.DataFrame(columns=CVCompare_columns)
row_index = 0

for k in [6.1]:#[k/10 for k in range(50,70)]:#[5]:#[1,2,3,4,5,6,7,8,9,10,11,16,17,18,20]: 
    start = time.perf_counter() 
    fi_threshold = k  # use k for that and iterate
    
    # Get the indices of all features over the importance threshold
    important_idx = np.where(feature_importance > fi_threshold)[0]
    # Create a list of all the feature names above the importance threshold
    important_features = np.array([feature_importance.keys()[x] for x in important_idx]) # change this in the others as well!
    print("\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):\n", 
            important_features)
    
    # Get the sorted indexes of important features
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    print("\nFeatures sorted by importance (DESC):\n", important_features[sorted_idx])
    
    #define new train set for CV loop with reduced columns according to feature importance threshold
    dtrain = xgboost.DMatrix(data[important_features[sorted_idx]], label = data[target].values, feature_names = important_features[sorted_idx], nthread = 8)
    
    print("CV with selected threshold of {}  now".format(k))
    # Run CV
    cv_results = xgboost.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        seed=seed,
        nfold=nfold,
        folds = folds,
        metrics=metrics,
        verbose_eval= verbose_eval,
        early_stopping_rounds=early_stopping_rounds
                )
    
    # get best RMSE
    results_train = cv_results['train-rmse-mean'].min()
    results_test = cv_results['test-rmse-mean'].min()
    results_test_std = cv_results['test-rmse-std'][cv_results['test-rmse-mean'].idxmin()]
    boost_rounds = cv_results['test-rmse-mean'].idxmin() + 1 
    
    # store results in DF
    CVCompare.loc[row_index, "threshold k"] = k
    CVCompare.loc[row_index, "RMSE base train"] = train_accuracy_base
    CVCompare.loc[row_index, "RMSE base test"] = val_accuracy_base
    CVCompare.loc[row_index, "CV RMSE train"] = results_train
    CVCompare.loc[row_index, "CV RMSE test"] = results_test
    CVCompare.loc[row_index, "CV RMSE test 3*STD"] = results_test_std*3
    CVCompare.loc[row_index, "CV boost_rounds"] = boost_rounds
    duration = time.perf_counter() - start
    print("\tRuntime of all CV folds for {} features was {:.0f}:{:.0f}:{:.1f}".format(k,
      duration // 3600, (duration % 3600 // 60), duration % 60))
    CVCompare.loc[row_index, "time"] = "{:.0f}:{:.0f}:{:.1f}".format(\
      duration // 3600, (duration % 3600 // 60), duration % 60)
    row_index+=1

# print and sort table:
CVCompare.sort_values(by= ["CV RMSE test"], ascending = True, inplace=True)
print(CVCompare)
"""
#%%
#results for the feature importances interaction
"""
   threshold k RMSE base train RMSE base test CV RMSE train CV RMSE test  \
4            5         2165.32        2514.21       2179.33      2499.05   
2            3         2165.32        2514.21       2152.03      2499.23   
5            6         2165.32        2514.21        2185.6      2500.58   
3            4         2165.32        2514.21       2184.66       2500.9   
6            7         2165.32        2514.21       2199.87      2502.23   
1            2         2165.32        2514.21       2172.43      2503.09   
0            1         2165.32        2514.21       2135.81      2503.76   
7            8         2165.32        2514.21       2149.99      2506.87   
10          11         2165.32        2514.21       2157.54      2543.59   
11          16         2165.32        2514.21       2157.54      2543.59   
9           10         2165.32        2514.21       2161.24      2553.75   
12          17         2165.32        2514.21        2191.1      2556.69   
8            9         2165.32        2514.21       2145.11      2565.28   
13          18         2165.32        2514.21       2648.16      2664.59   
14          20         2165.32        2514.21       2648.16      2664.59   

   CV RMSE test 3*STD CV boost_rounds       time  
4             236.646            1562  0:44:28.0  
2             239.584            1636   1:6:10.1  
5             238.605            1567  0:36:49.0  
3             236.309            1480  0:48:58.5  
6             239.143            1542  0:29:50.5  
1             244.831            1450  1:28:40.7  
0             238.243            1638    2:9:4.4  
7             234.861            2129  0:26:22.7  
10            271.081            3942  0:30:20.2  
11            271.081            3942   0:30:7.0  
9             267.887            3362  0:27:57.5  
12            264.589            4023  0:27:57.9  
8             270.263            3180  0:28:12.4  
13            243.634            2558  0:16:19.6  
14            243.634            2558  0:16:23.7  

   threshold k RMSE base train RMSE base test CV RMSE train CV RMSE test  \
11         6.1         2165.32        2514.21       2195.66      2511.97   
10           6         2165.32        2514.21       2195.66      2511.97   
14         6.4         2165.32        2514.21       2185.46      2512.26   
13         6.3         2165.32        2514.21       2186.38      2512.38   
12         6.2         2165.32        2514.21       2186.38      2512.38   
9          5.9         2165.32        2514.21       2167.14      2512.53   
7          5.7         2165.32        2514.21       2167.14      2512.53   
8          5.8         2165.32        2514.21       2167.14      2512.53   
15         6.5         2165.32        2514.21        2177.3      2512.81   
17         6.7         2165.32        2514.21        2177.3      2512.81   
16         6.6         2165.32        2514.21        2177.3      2512.81   
5          5.5         2165.32        2514.21       2195.51      2513.02   
6          5.6         2165.32        2514.21       2195.51      2513.02   
4          5.4         2165.32        2514.21       2171.56      2513.39   
3          5.3         2165.32        2514.21       2171.56      2513.39   
2          5.2         2165.32        2514.21       2171.56      2513.39   
1          5.1         2165.32        2514.21       2171.56      2513.39   
19         6.9         2165.32        2514.21       2204.54      2513.73   
18         6.8         2165.32        2514.21       2204.54      2513.73   
0            5         2165.32        2514.21       2163.83      2514.79  
"""

#%%
"""
# 31 parameters with best rmse for 5 folds
# we go with k = 6.1 for the features:
xgb_thresh = ['Product_ID_code',
 'Product_Category_1*User_ID_code',
 'Product_Category_1/User_ID_code',
 'Product_Category_2/User_ID_code',
 'Product_Category_2*User_ID_code',
 'Product_Category_3/User_ID_code',
 'Product_Category_3*User_ID_code',
 'User_ID_code*City_Category_code',
 'Occupation*User_ID_code',
 'Product_Category_1',
 'User_ID_code*Age_code',
 'Marital_Status/User_ID_code',
 'Product_Category_1-User_ID_code',
 'User_ID_code/City_Category_code',
 'User_ID_code',
 'User_ID_code*Stay_In_Current_City_Years_code',
 'Product_Category_1*Product_Category_2',
 'Product_Category_1+User_ID_code',
 'Marital_Status*User_ID_code',
 'Product_Category_2/Product_Category_3',
 'Occupation/User_ID_code',
 'User_ID_code/Age_code',
 'Product_Category_1/Product_Category_2',
 'Occupation-User_ID_code',
 'User_ID_code*Gender_code',
 'User_ID_code/Stay_In_Current_City_Years_code',
 'User_ID_code/Gender_code',
 'User_ID_code-Age_code',
 'Occupation+User_ID_code',
 'Product_Category_1*Product_Category_3',
 'Product_Category_2-User_ID_code']
"""
#%%
"""
# =============================================================================
# can also use PCA 
# =============================================================================
# need to apply scaling again for pca...
predictors = scaled_data.drop(target, axis=1).columns.values

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaled = scaler.fit(scaled_data[predictors])
scaled_data2 = scaled.transform(scaled_data[predictors])
scaled_data2 = pd.DataFrame(scaled_data2, columns = predictors )
scaled_data2 = pd.concat([scaled_data2, scaled_data[target]], axis=1)
#%%
scaled_data2.sample()
"""
#%%
"""
def plot_pca(data, predictors, iterated_power, n_components):
    pca_item = PCA(iterated_power=iterated_power, n_components= n_components)
    components = pca_item.fit(data[predictors])
    transformed = pca_item.transform(data[predictors])
    
    var = components.explained_variance_ratio_ #amount of variance that each PC explains
    var1 = np.cumsum(np.round(components.explained_variance_ratio_,3)) #cumulative variance explained

    number_points = np.arange(1,n_components+1)
    plt.figure(figsize=(12,6))
    sns.set_style("whitegrid", {'axes.grid' : True})
    plt.plot(number_points, var1, marker="o", markerfacecolor="r", markersize=10)
    for x,y in zip(number_points,var1):
        plt.annotate(str(round(y,2)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')
    plt.show() 
    print("\nIndividual variance explained by the components: ", var)
    print("\nCumulative variance explained by adding each new component: ", var1)
    return pd.DataFrame(transformed)

x_pca = plot_pca(scaled_data2, predictors, 7, 80)
# do cv with PCA 20, 60, 70, 80 components

#%%
# =============================================================================
# apply the pca and test with XGB
# =============================================================================
predictors = scaled_data2.drop(target, axis=1).columns.values
data = scaled_data2


CVCompare_columns = ["selected k", "RMSE train", "RMSE test", "RMSE test 3*STD", "boost_rounds", "time"]
CVCompare = pd.DataFrame(columns=CVCompare_columns)
row_index = 0

params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': .8,
    'colsample_bytree': .8,
    'nthread' : 8,
    # Other parameters
    'objective':'reg:linear',
    #'booster':'gblinear', # instead of gbtree for testing?
    'seed' : 42,
}
num_boost_round = 100000
early_stopping_rounds=400

# create lists to store train and validation CV scores after each full kfold step with all iterations
train_score_CV = []
val_score_CV = []
#create lists to store std scores for every iteration (all folds)
train_acc_std = []
val_acc_std = []

for k in [8,10,12,14,16,18, 30, 110]: #values for PCA
    #set up lists for results
    train_score = []
    val_score = []
    boost_rounds = []
    start = time.perf_counter()
    for n, (train_i, test_i) in enumerate(cv_split.split(data[predictors], data[target])): # use kfold and "average" over the whole dataset, use early stopping in xgboost.train for every eval_set
        print("\nfitting CV folds k = {}...".format(n+1))    
        X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
        y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]
                
        print("cv with {} components now".format(k))
        #scale data now! - fit on train, transform on both
#        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#        scaled = scaler.fit(X_train)
#        X_train = scaled.transform(X_train)
#        X_train = pd.DataFrame(X_train)
#        
#        # apply scaling to test set
#        X_val = scaled.transform(X_val)
#        X_val = pd.DataFrame(X_val)
        
        # do PCA
        pca_item = PCA(iterated_power=7, n_components= k)
        components = pca_item.fit(X_train)
        transformed = pca_item.transform(X_train)
        transformed_test = pca_item.transform(X_val)
        var1 = np.cumsum(np.round(components.explained_variance_ratio_,3)) #cumulative variance explained
        print("\nCumulative variance explained by adding each new component: ", var1)
        
        #DMatrix for every train and val set in folds
        dtrain = xgboost.DMatrix(transformed, label=y_train.values, feature_names = np.arange(1,k+1).astype(str) ,nthread = 8)
        dtest = xgboost.DMatrix(transformed_test, label=y_val.values, feature_names = np.arange(1,k+1).astype(str), nthread = 8)

        # fit the model only using that many training examples # test if using evals set and best iterations is ok
        reg = xgboost.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtest, "Test")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval = 300
                    )
        
        #print and store boost rounds
        print("Best RMSE: {:.2f} in {} rounds".format(reg.best_score, reg.best_iteration+1))  
        boost_rounds.append(reg.best_iteration)
        #calculate the training accuracy 
        trainpreds = reg.predict(dtrain,  ntree_limit=reg.best_ntree_limit)
        train_accuracy = np.sqrt(mean_squared_error(
	                                   y_train,
	                                    trainpreds)) 
        #calculate the validation accuracy 
        valpreds =  reg.predict(dtest, ntree_limit=reg.best_ntree_limit)
        val_accuracy = np.sqrt(mean_squared_error(
	                                    y_val,
	                                    valpreds
	                                    ))
   
        # store the scores in their respective lists
        train_score.append(train_accuracy)
        val_score.append(val_accuracy)
        
    #store mean boost rounds over folds
    boost_rounds = np.mean(boost_rounds)
    # store the stds of the cv fold values
    train_acc_std = (np.std(train_score))
    val_acc_std = (np.std(val_score))
    # store means of the individual scores to get final cv scores
    train_score_CV = (np.mean(train_score))
    val_score_CV = (np.mean(val_score))
    
    # store results in DF
    CVCompare.loc[row_index, "selected k"] = k
    CVCompare.loc[row_index, "RMSE train"] = train_score_CV
    CVCompare.loc[row_index, "RMSE test"] = val_score_CV
    CVCompare.loc[row_index, "RMSE test 3*STD"] = val_acc_std*3
    CVCompare.loc[row_index, "boost_rounds"] = boost_rounds
    duration = time.perf_counter() - start
    CVCompare.loc[row_index, "time"] = duration
    row_index+=1

# print and sort table:
CVCompare.sort_values(by= ["RMSE test"], ascending = True, inplace=True)
print(CVCompare)

# PCA delivers bad results. High RMSE... maybe need dummies for that?
"""
#%%
#"""
# Comparison of some models

# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
       #Ensemble Methods 
       #ensemble.AdaBoostRegressor(),
       #ensemble.BaggingRegressor(),
       #ensemble.ExtraTreesRegressor(),
       #ensemble.GradientBoostingRegressor(),
       ('rfr', ensemble.RandomForestRegressor()),
       
       
       #GLM
       ('lr', linear_model.LinearRegression()),
       #linear_model.RidgeCV(),
       #linear_model.LassoCV(),
       #linear_model.SGDRegressor(),
       ('elnet', linear_model.ElasticNetCV()),
       ('RANSAC', linear_model.RANSACRegressor()),
       
       #Nearest Neighbor
       ('knn', neighbors.KNeighborsRegressor()),
       
       #xgboost
       ('xgb', xgboost.XGBRegressor())
       ]

#"""
#%%
# =============================================================================
# we will define some functions we need later on here ... 
# for easier readability later
# =============================================================================
   #%%
#from matplotlib.ticker import FuncFormatter   
def plot_learning_curves_CV_reg(estimator, data = None, predictors= None, target = None,
	                         suptitle='', title='', xlabel='', ylabel='', ylabel2='', k =10, granularity = 10, train_size= 0.01, test_size = 0.01, split = "Shuffle", random_state = 42, scale = True, center = True):
    """
    Plots learning curves for a given estimator using the full train dataset and CV in this function
 
    Parameters
    ----------
	
    estimator : sklearn estimator
    data : pd.DataFrame
	        training set (raw data)
	    predictors : str of column names in dataframe
	        training set (without response)
	    target : str of column name in dataframe
	        ground truth (response)
	    suptitle : str
	        Chart suptitle # YES method is called fig.suptitle
	    title: str
	        Chart title
	    xlabel: str
	        Label for the X axis
	    ylabel: str
	        Label for the y axis
      k : int
          number of splits for the cv method 
     split: str
         method of cv. Options: Shuffle and KFold
     test_size : float (0, 1)
         defines the relative size for the test set used in cv method ShuffleSplit 
     granularity : int
         defines the number of incremental data sizes used to calculate the learning curve 
         
     Returns
	    -------
	
    Plot of learning curves
    """
    # create lists to store train and validation CV scores after each full kfold step with all iterations
    train_score_CV = []
    val_score_CV = []
    train_r2_score_CV = []
    val_r2_score_CV = []
    #create lists to store std scores for every iteration (all folds)
    train_acc_std = []
    val_acc_std = []
    train_r2_std = []
    val_r2_std = []

    # create the split strategy
    if split == "Shuffle":
        cv = ShuffleSplit(n_splits= k , train_size= train_size, test_size= test_size, random_state = random_state)  
        max_train_samples = len(data)*train_size 
    elif split == "KFold":
        cv = KFold(n_splits = k, shuffle=True, random_state = random_state)
        max_train_samples = len(data)-len(data)/k
    #max_train_samples = len(data) # can use max data indices because we shuffle the data and we can't limit indices or we lose the right hand side of our data
    
    # create ten incremental training set sizes
    training_set_sizes = np.linspace(15, max_train_samples, granularity, dtype='int')
    # for each one of those training set sizes do the steps
    for n, i in enumerate(training_set_sizes):
        print("fitting CV folds k = {}...".format(n+1))
        # create lists to store train and validation scores for each set of kfold subloops 
        train_score = []
        val_score = []
        train_r2_score = []
        val_r2_score = []
                
        for train_i, test_i in cv.split(data[predictors], data[target]): # use kfold to "average" over the whole dataset and compute a smoothed out learning curve
            X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
            y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]
            
            do_scale = 0
            
            if (scale == True) & (center == False):
                with_std = True
                with_mean = False
                do_scale = 1
            elif (scale == False) & (center == True):
                with_std = False
                with_mean = True
                do_scale = 1
            elif (scale == True) & (center == True):
                with_std = True
                with_mean = True
                do_scale = 1
                
            if do_scale == 1:
                
                #scale data now! - fit on train, transform on both
                scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
                scaled = scaler.fit(X_train)
                X_train = scaled.transform(X_train)
                X_train = pd.DataFrame(X_train)
                
                # apply scaling to test set
                X_val = scaled.transform(X_val)
                X_val = pd.DataFrame(X_val)

            # fit the model only using that many training examples
            ravel_y_train = np.array(y_train.iloc[0:i]).ravel()
            estimator.fit(X_train.iloc[0:i, :]
                    ,ravel_y_train) # ravel the vector
            
            #calculate the training accuracy only using those training examples
            train_accuracy = np.sqrt(mean_squared_error(
	                                   y_train.iloc[0:i],
	                                    estimator.predict(X_train.iloc[0:i, :])))
	                                    
            #calculate the validation accuracy using the whole validation set
            val_accuracy = np.sqrt(mean_squared_error(
	                                    y_val,
	                                    estimator.predict(X_val)
	                                    ))
            # calculate r2_score as well
            train_r2 = r2_score(y_train.iloc[0:i], estimator.predict(X_train.iloc[0:i, :]))
            val_r2 = r2_score(y_val, estimator.predict(X_val))
            # store the scores in their respective lists
            train_r2_score.append(train_r2)
            val_r2_score.append(val_r2)
        
            # store the scores in their respective lists
            train_score.append(train_accuracy)
            val_score.append(val_accuracy)

        # append the stds of the cv fold values
        train_acc_std.append(np.std(train_score))
        val_acc_std.append(np.std(val_score))
        train_r2_std.append(np.std(train_r2_score)) 
        val_r2_std.append(np.std(val_r2_score)) 
        # append means of the individual scores to get final cv scores
        train_score_CV.append(np.mean(train_score))
        val_score_CV.append(np.mean(val_score))
        train_r2_score_CV.append(np.mean(train_r2_score))
        val_r2_score_CV.append(np.mean(val_r2_score))  
    
    # plot learning curves on different charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(training_set_sizes, train_score_CV, c='gold', marker = "o")
    ax.plot(training_set_sizes, val_score_CV, c='steelblue', marker = "o")
    # plot the train / test score std ranges
    ax.fill_between(training_set_sizes, np.array(train_score_CV) - np.array(train_acc_std), np.array(train_score_CV) + np.array(train_acc_std), alpha = 0.15, color='gold')
    ax.fill_between(training_set_sizes, np.array(val_score_CV) - np.array(val_acc_std), np.array(val_score_CV) + np.array(val_acc_std), alpha = 0.15, color='steelblue')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(training_set_sizes, train_r2_score_CV, c='red', marker = "o")
    ax2.plot(training_set_sizes, val_r2_score_CV, c='green', marker = "o")
    ax2.fill_between(training_set_sizes, np.array(train_r2_score_CV) - np.array(train_r2_std), np.array(train_r2_score_CV) + np.array(train_r2_std), alpha = 0.15, color='red')
    ax2.fill_between(training_set_sizes, np.array(val_r2_score_CV) - np.array(val_r2_std), np.array(val_r2_score_CV) + np.array(val_r2_std), alpha = 0.15, color='green')
	
    # format the charts to make them look nice
    fig.suptitle(suptitle, fontweight='bold', fontsize='20')
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel, size=16)
    ax.set_ylabel(ylabel, size=16)
    ax.legend(['training set', 'validation set'], fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(min(train_score_CV[2:])-min(train_score_CV[2:])/2, max(val_score_CV[2:])+max(val_score_CV[2:])/10)
    ax.grid(b=True)
    for x,y in zip(training_set_sizes,val_score_CV):
        ax.annotate(str(round(y,3)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')
    
    fig2.suptitle(suptitle, fontweight='bold', fontsize='20')
    ax2.set_title(title, size=20)
    ax2.set_xlabel(xlabel, size=16)
    ax2.set_ylabel(ylabel2, size=16)
    ax2.legend(['training set', 'validation set'], fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(b=True)
    for x,y in zip(training_set_sizes,val_r2_score_CV):
        ax2.annotate(str(round(y,3)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')
	

    def percentages(x, pos):
        """The two args are the value and tick position"""
        if x < 1:
            return '{:1.0f}'.format(x*100)
        return '{:1.0f}%'.format(x*100)
	
    
    def numbers(x, pos):
         """The two args are the value and tick position"""
         if x >= 1000:
             return '{:1,.0f}'.format(x)
         return '{:1.0f}'.format(x)

    #x_formatter = FuncFormatter(numbers)
    #ax.xaxis.set_major_formatter(x_formatter)
    #ax2.xaxis.set_major_formatter(x_formatter)

    #y_formatter = FuncFormatter(percentages)
    #ax.yaxis.set_major_formatter(y_formatter)
    #ax2.yaxis.set_major_formatter(y_formatter)
   

#%%
def CV_target_max_purchase(estimator, data = None, predictors= None, target = None,
	                          percentile = None, percentile_remove = None, k = 10, train_size= 0.9, test_size = 0.1, split = "KFold", 
                              random_state = 42, scale = False, center = False):


    # created lists to store the individual results
    train_score = []
    val_score = []
    train_r2_score = []
    val_r2_score = []
    # create lists to store train and validation CV scores after each full kfold step with all iterations
    train_score_CV = []
    val_score_CV = []
    train_r2_score_CV = []
    val_r2_score_CV = []
    #create lists to store std scores for every iteration (all folds)
    train_acc_std = []
    val_acc_std = []
    train_r2_std = []
    val_r2_std = []

    # create the split strategy
    if split == "Shuffle":
        cv = ShuffleSplit(n_splits= k , train_size= train_size, test_size= test_size, random_state = random_state)  
    elif split == "KFold":
        cv = KFold(n_splits = k, shuffle=True, random_state = random_state)
    

    # iterate through the folds and fit / predict on train/val folds
    for n, (train_i, test_i) in enumerate(cv.split(data[predictors], data[target])): # use kfold to "average" over the whole dataset and compute a smoothed out learning curve
        print("fitting CV folds k = {}...".format(n+1))
        X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
        y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]
        X_train = X_train.astype(float)
        X_val = X_val.astype(float)
        y_train = y_train.astype(float)
        y_val = y_val.astype(float)
        
        if percentile != None:
            print("we change max_purchase")
            # get the indices for max value percentile
            max_purchase = np.percentile(y_train, percentile) # we set 99.9 percentile as upper boundary for purchase amount and check change in rmse
            if percentile_remove == "remove":
                print("we remove samples above or equal max_purchase from train")
                # return only the indices that have targets smaller than max_purchase
                X_train = X_train.loc[y_train <= max_purchase]
                y_train = y_train.loc[y_train <= max_purchase]
            else:
                # apply the max value percentile to target in train only
                print("applying max_purchase percentile to train")
                y_train.loc[y_train > max_purchase] = max_purchase
                

            
        do_scale = 0
        if (scale == True) & (center == False):
            with_std = True
            with_mean = False
            do_scale = 1
        elif (scale == False) & (center == True):
            with_std = False
            with_mean = True
            do_scale = 1
        elif (scale == True) & (center == True):
            with_std = True
            with_mean = True
            do_scale = 1
            
        if do_scale == 1:
            
            #scale data now! - fit on train, transform on both
            scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
            scaled = scaler.fit(X_train)
            X_train = scaled.transform(X_train)
            X_train = pd.DataFrame(X_train)
            
            # apply scaling to test set
            X_val = scaled.transform(X_val)
            X_val = pd.DataFrame(X_val)

        # fit the model with all training folds and predict val folds
        ravel_y_train = np.array(y_train[:]).ravel()
        estimator.fit(X_train
                ,ravel_y_train) # ravel the vector

        #calculate the training accuracy using the whole train set
        train_accuracy = np.sqrt(mean_squared_error(
	                                   y_train,
	                                    estimator.predict(X_train)))

        #calculate the validation accuracy using the whole validation set
        val_accuracy = np.sqrt(mean_squared_error(
	                                    y_val,
	                                    estimator.predict(X_val)
	                                    ))
        # calculate r2_score as well
        train_r2 = r2_score(y_train, estimator.predict(X_train))
        val_r2 = r2_score(y_val, estimator.predict(X_val))
        # store the scores in their respective lists
        train_r2_score.append(train_r2)
        val_r2_score.append(val_r2)
    
        # store the scores in their respective lists
        train_score.append(train_accuracy)
        val_score.append(val_accuracy)

    # append the stds of the cv fold values
    train_acc_std.append(np.std(train_score))
    val_acc_std.append(np.std(val_score))
    train_r2_std.append(np.std(train_r2_score)) 
    val_r2_std.append(np.std(val_r2_score)) 
    # append means of the individual scores to get final cv scores
    train_score_CV.append(np.mean(train_score))
    val_score_CV.append(np.mean(val_score))
    train_r2_score_CV.append(np.mean(train_r2_score))
    val_r2_score_CV.append(np.mean(val_r2_score)) 
    
    return train_score_CV, val_score_CV, train_r2_score_CV, val_r2_score_CV

   #%%
   
# =============================================================================
# XGB learning curves for native api
# =============================================================================
   
def plot_learning_curves_XGB_reg(params, data = None, predictors= None, target = None,
	                         suptitle='', title='', xlabel='', ylabel='', ylabel2='', k =10, granularity = 10, train_size= 0.01, test_size = 0.01, split = "Shuffle", random_state = 42, scale = True, center = True, num_boost_round = 100000, early_stopping_rounds = 400):
    # create lists to store train and validation CV scores after each full kfold step with all iterations
    train_score_CV = []
    val_score_CV = []
    train_r2_score_CV = []
    val_r2_score_CV = []
    #create lists to store std scores for every iteration (all folds)
    train_acc_std = []
    val_acc_std = []
    train_r2_std = []
    val_r2_std = []

    # create the split strategy
    if split == "Shuffle":
        cv = ShuffleSplit(n_splits= k , train_size= train_size, test_size= test_size, random_state = random_state)  
        max_train_samples = len(data)*train_size 
    elif split == "KFold":
        cv = KFold(n_splits = k, shuffle=True, random_state = random_state)
        max_train_samples = len(data)-len(data)/k
    #max_train_samples = len(data) # can use max data indices because we shuffle the data and we can't limit indices or we lose the right hand side of our data
    
    # create ten incremental training set sizes
    training_set_sizes = np.linspace(15, max_train_samples, granularity, dtype='int')
    # for each one of those training set sizes do the steps
    for n, i in enumerate(training_set_sizes):
        print("fitting CV folds k = {}...".format(n+1))
        # create lists to store train and validation scores for each set of kfold subloops 
        train_score = []
        val_score = []
        train_r2_score = []
        val_r2_score = []
        
        
        for train_i, test_i in cv.split(data[predictors], data[target]): # use kfold to "average" over the whole dataset and compute a smoothed out learning curve
            X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
            y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]
            
            #assign xgb DMatrix
            dtrain = xgboost.DMatrix(X_train.iloc[0:i, :].values, label=y_train.iloc[0:i].values, nthread = 8)
            dtest = xgboost.DMatrix(X_val.values, label=y_val.values, nthread = 8)
            
            
            do_scale = 0
            
            if (scale == True) & (center == False):
                with_std = True
                with_mean = False
                do_scale = 1
            elif (scale == False) & (center == True):
                with_std = False
                with_mean = True
                do_scale = 1
            elif (scale == True) & (center == True):
                with_std = True
                with_mean = True
                do_scale = 1
                
            if do_scale == 1:
                
                #scale data now! - fit on train, transform on both
                scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
                scaled = scaler.fit(X_train)
                X_train = scaled.transform(X_train)
                X_train = pd.DataFrame(X_train)
                
                # apply scaling to test set
                X_val = scaled.transform(X_val)
                X_val = pd.DataFrame(X_val)

            # fit the model only using that many training examples # test if using evals set and best iterations is ok
            reg = xgboost.train(
                        params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=[(dtest, "Test")],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval = 200
                        )
            #calculate the training accuracy only using those training examples
            trainpreds = reg.predict(dtrain,  ntree_limit=reg.best_ntree_limit)
            train_accuracy = np.sqrt(mean_squared_error(
	                                   y_train.iloc[0:i],
	                                    trainpreds)) 
            #calculate the validation accuracy using the whole validation set
            valpreds =  reg.predict(dtest, ntree_limit=reg.best_ntree_limit)
            val_accuracy = np.sqrt(mean_squared_error(
	                                    y_val,
	                                    valpreds
	                                    ))
            # calculate r2_score as well
            train_r2 = r2_score(y_train.iloc[0:i], trainpreds)
            val_r2 = r2_score(y_val, valpreds)
            # store the scores in their respective lists
            train_r2_score.append(train_r2)
            val_r2_score.append(val_r2)
        
            # store the scores in their respective lists
            train_score.append(train_accuracy)
            val_score.append(val_accuracy)

        # append the stds of the cv fold values
        train_acc_std.append(np.std(train_score))
        val_acc_std.append(np.std(val_score))
        train_r2_std.append(np.std(train_r2_score)) 
        val_r2_std.append(np.std(val_r2_score)) 
        # append means of the individual scores to get final cv scores
        train_score_CV.append(np.mean(train_score))
        val_score_CV.append(np.mean(val_score))
        train_r2_score_CV.append(np.mean(train_r2_score))
        val_r2_score_CV.append(np.mean(val_r2_score))  
    
    # plot learning curves on different charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(training_set_sizes, train_score_CV, c='gold', marker = "o")
    ax.plot(training_set_sizes, val_score_CV, c='steelblue', marker = "o")
    # plot the train / test score std ranges
    ax.fill_between(training_set_sizes, np.array(train_score_CV) - np.array(train_acc_std), np.array(train_score_CV) + np.array(train_acc_std), alpha = 0.15, color='gold')
    ax.fill_between(training_set_sizes, np.array(val_score_CV) - np.array(val_acc_std), np.array(val_score_CV) + np.array(val_acc_std), alpha = 0.15, color='steelblue')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(training_set_sizes, train_r2_score_CV, c='red', marker = "o")
    ax2.plot(training_set_sizes, val_r2_score_CV, c='green', marker = "o")
    ax2.fill_between(training_set_sizes, np.array(train_r2_score_CV) - np.array(train_r2_std), np.array(train_r2_score_CV) + np.array(train_r2_std), alpha = 0.15, color='red')
    ax2.fill_between(training_set_sizes, np.array(val_r2_score_CV) - np.array(val_r2_std), np.array(val_r2_score_CV) + np.array(val_r2_std), alpha = 0.15, color='green')
	
    # format the charts to make them look nice
    fig.suptitle(suptitle, fontweight='bold', fontsize='20')
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel, size=16)
    ax.set_ylabel(ylabel, size=16)
    ax.legend(['training set', 'validation set'], fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(min(train_score_CV[2:])-min(train_score_CV[2:])/2, max(val_score_CV[2:])+max(val_score_CV[2:])/10)
    ax.grid(b=True)
    for x,y in zip(training_set_sizes,val_score_CV):
        ax.annotate(str(round(y,3)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')
    
    fig2.suptitle(suptitle, fontweight='bold', fontsize='20')
    ax2.set_title(title, size=20)
    ax2.set_xlabel(xlabel, size=16)
    ax2.set_ylabel(ylabel2, size=16)
    ax2.legend(['training set', 'validation set'], fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(b=True)
    for x,y in zip(training_set_sizes,val_r2_score_CV):
        ax2.annotate(str(round(y,3)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')
	

    def percentages(x, pos):
        """The two args are the value and tick position"""
        if x < 1:
            return '{:1.0f}'.format(x*100)
        return '{:1.0f}%'.format(x*100)
	
    
    def numbers(x, pos):
         """The two args are the value and tick position"""
         if x >= 1000:
             return '{:1,.0f}'.format(x)
         return '{:1.0f}'.format(x)




#%%
# =============================================================================
# # classes for the pipeline
# =============================================================================
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **fit_params):
        self.columns = columns
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        return X[self.columns] # returns selected columns for the preprocessing inside pipeline
#test  
#ColumnSelector(["Gender", 'City_Category']).fit_transform(train)
#%%
# could write custom imputer also for the category columns
class CustomImp(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None, value = None, **fit_params):
        self.columns = columns # impute NaN in columns with specified value
        self.value = value
        
    def fit(self, X, y=None):
        return self # not relevant here
    
    def transform(self, X, y=None):
        output = X.copy(deep = True)
        if self.columns == None:
            for colname,col in output.iteritems(): 
                output[colname] = X[colname].fillna(self.value)
            return output
        
        else:
            for colname in self.columns:
                output[colname] = X[colname].fillna(self.value)
            return output[self.columns]

       
#Test
#CustomImp(["Product_Category_2", "Product_Category_3"], value = -1).fit_transform(train)
#%%
class LabelEncoderPipe(LabelEncoder):
    def __init__(self, columns = None):
        self.columns = columns # array of column names to encode
        
    def fit(self, X, y=None):
        return self # not relevant here
    
    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using LabelEncoder(). If no columns
        are specified, transforms all columns in X
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)

#Testcase:  
"""
pd.set_option("display.max_columns",13)
pipe=Pipeline([
        ('sel_cat', ColumnSelector(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])),
        ('enc', LabelEncoderPipe(#columns= "User_ID" # only need to specify columns if we want specific ones. But we specify with ColumnSelector already, so no need.
                ))
])
pipe.fit_transform(train)
"""

#%%
data = data1
predictors = data1_code
# compare learning curves for some regressors:
for name,model in MLA:
    plot_learning_curves_CV_reg(model, data, predictors, target, suptitle = "learning curve tests", title="Base model CV scores for model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.1, test_size = 0.1, split = "Shuffle", scale = False, center = False, random_state = 42 )
#xgb seems to be a very sensible choice already
#linear models dont work well with the current dataset structure .... getting dummies far improves rmse
# but we don't explore that approach here further
#%%

#create table to compare MLA metrics
MLA_columns = ["MLA Name", "MLA Train  RMSE Mean", "MLA Test  RMSE Mean", "MLA Test  RMSE 3*STD", "MLA TIME"]
MLA_compare = pd.DataFrame(columns=MLA_columns)
# index through MLA and save performance to table
row_index = 0
for name, alg in MLA:
    #set name and params
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, "MLA Name"] = MLA_name
  
    #score model with cross validation
    cv_results = model_selection.cross_validate(alg, data[predictors], data[target], cv=cv_split, scoring = "neg_mean_squared_error", return_train_score=True, n_jobs = -1, verbose = 10)

    MLA_compare.loc[row_index, "MLA TIME"] = cv_results["fit_time"].mean()
    MLA_compare.loc[row_index, "MLA Train  RMSE Mean"] = np.sqrt(cv_results["train_score"].mean()*(-1))
    MLA_compare.loc[row_index, "MLA Test  RMSE Mean"] = np.sqrt(cv_results["test_score"].mean()*(-1))
    #if this is a non-bias random sample, then +/-3 std from the mean, should statistically capture 99,7% of the subsets
    MLA_compare.loc[row_index, "MLA Test  RMSE 3*STD"] = np.sqrt(cv_results["test_score"].std()*(3))
    #let's know the worst that can happen
    
    row_index+=1
# print and sort table:
MLA_compare.sort_values(by= ["MLA Test  RMSE Mean"], ascending = False, inplace=True)
MLA_compare
#%%
# plot the results
sns.barplot(x="MLA Test  RMSE Mean", y="MLA Name", data=MLA_compare, color="m")
plt.title("MLA RMSE Score\n")
plt.xlabel("RMSE Score")
plt.ylabel("Algorithm")
#%%
# =============================================================================
# Check XGB estimators number - prevent overfitting
# =============================================================================
"""
data = data1
#predictors = pred_final
#predictors = pred_test
predictors = data1_code
# Function for tuning XGB hyperparameters and visualizing results
# y defaults to our defined target variable values.

def modelfit_reg(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, folds = None, early_stopping_rounds=100, y=target):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgboost.DMatrix(dtrain[predictors].values, feature_names=predictors, label=y.values)
        cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds, stratified = False, verbose_eval= True, seed = 42, folds = folds)
        alg.set_params(n_estimators= int(cvresult.shape[0]/(1-1/folds.get_n_splits())))
        print("\nBest iteration: ",cvresult.shape[0], cvresult['test-rmse-mean'][cvresult.shape[0]-1])
        print("optimal number of estimators is: ", int(cvresult.shape[0]/(1-1/folds.get_n_splits())))
        
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y ,eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    ## insert printing for plot ##
    results = cvresult
    epochs = len(results["train-rmse-mean"])
    x_axis = range(0, epochs)
    
    # plot regression error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['train-rmse-mean'], label='Train')
    ax.plot(x_axis, results['test-rmse-mean'], label='Test')
    ax.set_ylim((2200, 2700))
    ax.legend()
    plt.ylabel('rmse')
    plt.title('XGBoost rmse')
    plt.show()
        
    #Print model report:
    print("\nModel Report")
    print("Fitted alg pred RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, dtrain_predictions))) 
    print("explained variance score: %f" % metrics.explained_variance_score(y, dtrain_predictions))
    print(pd.Series(alg.get_booster().get_score(importance_type='weight')).sort_values(ascending=False))
    feat_imp = pd.Series(alg.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    pyl.ylabel('Feature Importance Score')
    
    #native plotting of importance:
    plt.figure(figsize=(20,15))
    xgboost.plot_importance(alg, ax=plt.gca())

    
model = XGBRegressor(n_estimators=10000,
         n_jobs= 8,
         random_state = 42)
modelfit_reg(model, data,predictors,folds=cv_split, y=data[target])
#%%
# cross validate imputation of columns cat 2 and 3 and removal
# Loop with cross validate and feature union, preprocessing and pipeline
predictors = data1_code

modes_to_check= ["removal", "no_imputation", "imputation_0", "imputation_-1"]

CVCompare_columns = ["mode", "RMSE train", "RMSE test", "RMSE test 3*STD", "time"]
CVCompare = pd.DataFrame(columns=CVCompare_columns)
row_index = 0
seed = 42

cachedir = mkdtemp()

for mode in modes_to_check:
    if mode == "imputation_0":
        step = [('pre_cat', Pipeline([ # process categorical columns
                ('sel_cat', ColumnSelector(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])),
                ('enc', LabelEncoderPipe()) #columns= "User_ID" ==> only need to specify columns if we want specific ones. But we specify with ColumnSelector already, so no need.
                #('1hot', OneHotEncoder())
            ])),
            
            ('pre_nan', Pipeline([ # process our nan columns
                    ('sel_nominal', ColumnSelector(['Product_Category_2', 'Product_Category_3'])),
                    ('cust_imp', CustomImp(value = 0))
                        #('impute', Imputer(strategy = "median")),
                        # should use SimpleImputer and strategy = "category"...? from sklearn.impute
                        # might also influence error: MissingIndicator
                    ])),
    
            ('pre_none', Pipeline([ # add Columns as is
                                    ('sel_nominal', ColumnSelector(['Occupation', 'Product_Category_1', 'Marital_Status']))
                            ]))]
    
    elif mode == "imputation_-1":
            step = [('pre_cat', Pipeline([ # process categorical columns
                    ('sel_cat', ColumnSelector(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])),
                    ('enc', LabelEncoderPipe()) #columns= "User_ID" ==> only need to specify columns if we want specific ones. But we specify with ColumnSelector already, so no need.
                    #('1hot', OneHotEncoder())
                ])),
                
                ('pre_nan', Pipeline([ # process our nan columns
                        ('sel_nominal', ColumnSelector(['Product_Category_2', 'Product_Category_3'])),
                        ('cust_imp', CustomImp(value = -1))
                        ])),
        
                ('pre_none', Pipeline([ # add Columns as is
                                        ('sel_nominal', ColumnSelector(['Occupation', 'Product_Category_1', 'Marital_Status']))
                                ]))]  
    
    elif mode == "removal":
        step = [('pre_cat', Pipeline([ # process categorical columns
                ('sel_cat', ColumnSelector(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])),
                ('enc', LabelEncoderPipe()) #columns= "User_ID" ==> only need to specify columns if we want specific ones. But we specify with ColumnSelector already, so no need.
                ])),
        # leave out cat 2 and 3
                ('pre_none', Pipeline([ # add Columns as is
                        ('sel_nominal', ColumnSelector(['Occupation', 'Product_Category_1', 'Marital_Status']))
                            ]))]
        
    else:
        step = [('pre_cat', Pipeline([ # process categorical columns
                ('sel_cat', ColumnSelector(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])),
                ('enc', LabelEncoderPipe()) #columns= "User_ID" ==> only need to specify columns if we want specific ones. But we specify with ColumnSelector already, so no need.
                ])),
        
                ('pre_nan', Pipeline([ # add our nan columns unaltered
                        ('sel_nan', ColumnSelector(['Product_Category_2', 'Product_Category_3'])),
                        ])),
            
                ('pre_none', Pipeline([ # add Columns as is
                        ('sel_nominal', ColumnSelector(['Occupation', 'Product_Category_1', 'Marital_Status']))
                            ]))]
    
    
    features = Pipeline([
        ('union', FeatureUnion( 
                # feature union can be part of a pipeline itself
                ## This estimator applies a list of transformer objects in parallel to the input data, then concatenates the results 
                ## This is useful to combine several feature extraction mechanisms into a single transformer

                transformer_list= # add list of Pipeline-tuples in transformer_list for concatenation in FeautureUnion
                                 step,
                                transformer_weights = None
                )),
        #('select_best', SelectKBest(k=k, score_func = f_regression)), # select k best features - univariate selection. 
        #('standardize', StandardScaler()),
        ('xgb',  XGBRegressor(n_estimators=3000, #tradeoff between time and rmse
                             n_jobs= 8,
                             random_state = 42))
        ], memory = cachedir
        )

    test_pipe = features  

    # splitting strategy defined outside or in here
    #kfold = KFold(n_splits = 5, shuffle = True, random_state=seed)
    #ssplit = ShuffleSplit(n_splits = 5, test_size = 0.05, train_size= 0.1, random_state =seed)
    # evaluate 
    print("cv with {} now".format(mode))
    start = time.perf_counter()
    cv_pipe = model_selection.cross_validate(test_pipe, train[predictors], train[target], cv=cv_split, scoring= ("r2", "neg_mean_squared_error"), return_train_score=True, verbose = 10)
    
    # store results
    CVCompare.loc[row_index, "mode"] = mode
    CVCompare.loc[row_index, "RMSE train"] = np.sqrt(np.mean(-cv_pipe["train_neg_mean_squared_error"]))
    CVCompare.loc[row_index, "RMSE test"] = np.sqrt(np.mean(-cv_pipe["test_neg_mean_squared_error"]))
    CVCompare.loc[row_index, "RMSE test 3*STD"] = np.sqrt(3*np.std(cv_pipe["test_neg_mean_squared_error"]))
    duration = time.perf_counter() - start
    CVCompare.loc[row_index, "time"] = duration
    row_index+=1
    
    
    print("results for pipeline with mode = {} are: ".format(mode))
    print("rmse train: %.6g" % np.sqrt(np.mean(cv_pipe["train_neg_mean_squared_error"])*-1))
    print("rmse test: %.6g" % np.sqrt(np.mean(cv_pipe["test_neg_mean_squared_error"])*-1))
    print("elapsed time: %.6gs" % duration)
    
# print and sort table:
CVCompare.sort_values(by= ["RMSE test"], ascending = True, inplace=True)
print(CVCompare)
print("done with columns: ", predictors)

# Clear cache of Pipeline! 
rmtree(cachedir)
 #           mode RMSE train RMSE test RMSE test 3*STD     time
#1  no_imputation    2366.43    3041.3         608.733  1398.47 # no imputation works best, almost no difference though
#2   imputation_0    2370.16   3041.94         593.322  1445.32 # imputation of -1 or 0 is also ok though. Good for other algorithms that need imputation for stacking later!
#3  imputation_-1    2370.17   3041.95         591.485  1416.86 
#0        removal    2379.63   3155.15         634.414   1323.2 # we don't remove the columns! Removing them hurts submit RMSE by 2$ for me.

"""
#%%
"""
# =============================================================================
# ### Cleaning of outliers based on Squared Errors ###
# =============================================================================

# compare model performance before and after that step and see if test score improved #

def outlierCleaner(predictions, data, predictors, target, deletefraction = 0.01):
    
#         Clean away the fraction% of points that have the largest
#         residual errors (difference between the prediction
#         and the actual net worth).
#
#         Return a DataFrame named cleaned_data where
#         a column error was inserted with the computed squared errors 
#         between predictions and target values.

     cleaned_data = data.copy(deep = True)
     
     for i in range(len(predictions)):
         cleaned_data["errors"] = (cleaned_data[target].iloc[i]-predictions[i])**2
     print("sorting now")
     # remove the fraction of biggest errors (sort: descending)
     cleaned_data.sort_values(by = "errors", kind = "mergesort", ascending=False, inplace=True) 
     return cleaned_data.iloc[int((cleaned_data).shape[0]*deletefraction):]
"""
#%%
"""
# baseline:
predictors = data1_code
reg = XGBRegressor(n_estimators=3000, #tradeoff between time and rmse
                             n_jobs= 8,
                             random_state = 42)
reg.fit(data1[predictors], data1[target])
predictions = reg.predict(data1[predictors])
print("\nRMSE for full train set: ",np.sqrt(mean_squared_error(data1[target], predictions)))
# RMSE 2917
#%%
#should do this with CV loop ideally, but takes a long time as well.

cleaned_data = outlierCleaner(predictions, data1, predictors, target, deletefraction = 0.01)

reg2 = XGBRegressor(n_estimators=3000, #tradeoff between time and rmse
                             n_jobs= 8,
                             random_state = 42)
reg2.fit(cleaned_data[predictors], cleaned_data[target])
predictions2 = reg.predict(cleaned_data[predictors])
print("\nRMSE for train set w/o outliers: ",np.sqrt(mean_squared_error(cleaned_data[target], predictions2)))
# RMSE 2930 - removal does not help 
#we'll not explore this any further
"""
#%%
# large outliers in purchase? check!
print(np.percentile(data1["Purchase"], 99.9))
#%%
# visual check
plt.figure(figsize=(16,12))
plt.plot([x for x in range(0, train.shape[0])],train[target])
plt.ylim(23500, 24000)
# seems we have a few values above 23729, those don't seem huge. 
#%%
biggest_amounts = data1["Purchase"][data1[target] > np.percentile(data1["Purchase"], 99.9)].count() # 150 rows with Purchase amount bigger than 23900
print("\nThe biggest amounts number {} and those are {}% of the data.".format(biggest_amounts, round(biggest_amounts*100/data1.shape[0],3)))
#%%
# =============================================================================
# Test out effects of tinkering with a max_purchase value ceiling for Purchase. 
# Removing or changing the values will be explored. 
# =============================================================================

    #using the same model here for camparability of results
    # final model is tuned and has different hyperparameters
data = data1
predictors = data1_code
target = "Purchase"

model = XGBRegressor(n_estimators=3000, #tradeoff between time and rmse
                             n_jobs= 8,
                             random_state = 42)
# not removing anything
train_rmse, test_rmse, train_r2, test_r2 = CV_target_max_purchase(model, data, predictors, target, percentile = None,  k = 5)
values = [train_rmse, test_rmse, train_r2, test_r2]
prints = ["Train RMSE", "Test RMSE", "Train r2", "Test r2"]
results = pd.DataFrame(values, columns=["RMSE"], index = prints)
print(results)

#Result for  Train RMSE  is:  [2247.030819899587]
#Result for  Test RMSE  is:  [2477.5327876467727]
#Result for  Train r2  is:  [0.7998836356038319]
#Result for  Test r2  is:  [0.7567108355882443]
#%%
data = data1

#looking at the target value percentile > 99.9
# set the value of percentile 99.9 as max value for anything bigger
train_rmse, test_rmse, train_r2, test_r2 = CV_target_max_purchase(model, data, predictors, target, percentile = 99.9, k = 5)
values = [train_rmse, test_rmse, train_r2, test_r2]
prints = ["Train RMSE", "Test RMSE", "Train r2", "Test r2"]
results = pd.DataFrame(values, columns=["RMSE"], index = prints)
print(results)

#Result for  Train RMSE  is:  [2246.6903591665573]
#Result for  Test RMSE  is:  [2477.6498094371846] #worse on predictions -> we don't use the max_purchase ceiling. 
#                                                 BTW on submissions it also fares worse, around 2$ in RMSE
#Result for  Train r2  is:  [0.7999178077144078]
#Result for  Test r2  is:  [0.7566886140057932]
#%%
data = data1

predictors = data1_code
target = "Purchase"

# cv results for removing the max samples percentile > 99.9 from the set altogether:
train_rmse, test_rmse, train_r2, test_r2 = CV_target_max_purchase(model, data, predictors, target, percentile = 99.0, percentile_remove = "remove",  k = 5)
values = np.round(np.array([train_rmse, test_rmse, train_r2, test_r2]),2)
prints = ["Train RMSE", "Test RMSE", "Train r2", "Test r2"]
results = pd.DataFrame(values, columns=["RMSE"], index = prints)
print(results)
#Result for  Train RMSE  is:  [2244.9647446821191]
#Result for  Test RMSE  is:  [2476.8633793454364] # we seem to have a little improvement in test RMSE and r2 
# compared to the non-removed results of the 99.9 percentile 
 #---->   try a submission with removing the 99.9 percentile for Purchase value from train samples 
 # ---> Results: RMSE a little lower, like half a doller
#Result for  Train r2  is:  [0.79875511416184541]
#Result for  Test r2  is:  [0.756842924928679]
 
# try removing more than just 0.001, remove 0.01 and 0.1 and test RMSE:
# percentile = 99.0:
#                RMSE
#Train RMSE  2214.81
#Test RMSE   2504.85 # ok we did worse with 0.01 already. No need to look at this further.
#Train r2       0.79
#Test r2        0.75
 #%%

# using xgboost native api and cv for gridsearch

random_state = 42
target = "Purchase"
#data = scaled_data_vif#.iloc[0:10000,:]
#data = scaled_data#.iloc[0:10000,:]

#predictors = data.drop(target, axis=1).columns
#predictors = xgb_thresh

data = data1#.iloc[0:20000,:]
predictors = data1_code

# Hold back part of train set and split into train and val set
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=0.1, shuffle = True, random_state=random_state)

# use best features by xgb feature selection method

dtrain = xgboost.DMatrix(X_train, label=y_train, feature_names = predictors, nthread = 8)
dtest = xgboost.DMatrix(X_test, label=y_test, feature_names = predictors, nthread = 8)

#native xgb api params
# split in train test, keep eval_set back
# cv on training set, test on eval_set
params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'nthread' : 8,
    # Other parameters
    'objective':'reg:linear',
    #'booster':'gblinear', # instead of gbtree for testing?
    'seed' : random_state,
}
num_boost_round = 100000
#%%
"""
model = xgboost.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=300
)
print("optimal number of estimators is: ", int(model.best_ntree_limit/(1-0.1))) # eval set is 0.1
print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
#%%
plt.figure(figsize=(20,15))
xgboost.plot_importance(model, ax=plt.gca())
"""
#%%
"""
# You can try wider intervals with a larger step between
# each value and then narrow it down. Here after several
# iteration I found that the optimal value was in the
# following ranges.
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,7)
    for min_child_weight in range(1,31,6)
]
#%%
import time
# Define initial best params and RMSE
def xgb_native_gridsearch(dtrain, params, num_boost_round=100000, seed=42, nfold=5, folds=None, 
                          metrics = {'rmse'}, early_stopping_rounds = 100, gridsearch_params=None, parameter_pairs = ("max_depth", "min_child_weight"),
                          verbose_eval = True):
    results = []
    best_results = []
    starttotal = time.perf_counter()
    min_target = float("Inf")
    best_params = None
    firstdim = len(gridsearch_params)
    snddim = len(gridsearch_params[0])
    numberfolds = folds.get_n_splits()
    print("\nWe will fit {} parameter sets of size {} each for {} folds each, equal to {} total fits".format(firstdim, 
          snddim , numberfolds, numberfolds*firstdim)) # only firstdim in multi, since we fit every tuple for numberfolds folds
    for par1, par2 in gridsearch_params:
        startcv = time.perf_counter()
        print("\n\tCV with {}={} and {}={}".format(parameter_pairs[0], par1, parameter_pairs[1], par2))
        
        # Update our parameters
        params[parameter_pairs[0]] = par1
        params[parameter_pairs[1]] = par2
        
        # Run CV
        cv_results = xgboost.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            seed=seed,
            nfold=nfold,
            folds = folds,
            metrics=metrics,
            verbose_eval= verbose_eval,
            early_stopping_rounds=early_stopping_rounds
        )
        
        # Update best RMSE
        mean_target = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].idxmin() + 1 
        if mean_target < min_target:
            min_target = mean_target
            best_params = (par1,par2)
            best_results = cv_results
        print("\n\tRMSE {} for {} rounds".format(mean_target, boost_rounds))
        # store the results
        results.append(cv_results)
        # get the cv time for total n folds with parameter set
        endcv = time.perf_counter() - startcv  
        print("\tRuntime of all CV folds for parameters {}={} and {}={} was {:.0f}:{:.0f}:{:.0f}".format(parameter_pairs[0], par1, parameter_pairs[1], par2,
              endcv // 3600, (endcv % 3600 // 60), endcv % 60))
    
    
    ## insert printing for plot ## plot for best parameters
    epochs = len(best_results["train-rmse-mean"])
    x_axis = range(0, epochs)
    
    # plot regression error
    fig, ax = plt.subplots( figsize = (16,12))
    ax.plot(x_axis, best_results['train-rmse-mean'], label='Train')
    ax.plot(x_axis, best_results['test-rmse-mean'], label='Test')
    ax.set_ylim((2200, 2600))
    ax.legend()
    plt.ylabel('rmse')
    plt.title('XGBoost rmse')
    plt.show()

    # print total search runtime and best params / number of trees
    endtotal = time.perf_counter() - starttotal
    print("\nBest params:  {}={}, {}={}, RMSE: {}".format(parameter_pairs[0], best_params[0], 
          parameter_pairs[1], best_params[1], min_target))
    print('Total runtime is {:.0f}:{:.0f}:{:.0f}'.format(endtotal // 3600,
          (endtotal % 3600 // 60), endtotal % 60))
    return results # return all results over whole parameter search space

    #%%
cv_results = xgb_native_gridsearch(dtrain=dtrain, params=params, num_boost_round = 100000, folds=cv_split, gridsearch_params = gridsearch_params, parameter_pairs = ("max_depth", "min_child_weight"), verbose_eval = 400, early_stopping_rounds = 100)
#Best params for original dataset:  max_depth=4, min_child_weight=13, RMSE: 2489.2255859999996

#Best params for the interaction dataset with k = 39 features --> xgb_thresh: (k=5)
#Best params:  max_depth=3, min_child_weight=26, RMSE: 2497.0881103

#Best params for the interaction dataset with k = 31 features --> xgb_thresh: (k=6.1)
#Best params:  max_depth=3, min_child_weight=25, RMSE: 2496.3471191  <--- still higher than original dataset

#we go with the original dataset...

#%%
def xgb_gridserach_results_plot_(cv_results, gridsearch_params):
    # print results
    means = [cv_results[x]['test-rmse-mean'] for x in range(len(cv_results))]
    stds =  [cv_results[x]['test-rmse-std'] for x in range(len(cv_results))]
    params = gridsearch_params
    for mean, stdev, param in zip(means, stds, params):
        print("min test mean rmse: {:.2f}+{:.2f}std with params max_depth={} and min_child_weight={}".format(mean.min(),
              stdev[mean.idxmin()], param[0], param[1]))
        
    # plot results
    plt.figure(figsize=(20,16))
    for i, value in enumerate(gridsearch_params):
        plt.plot(cv_results[i].index.values, cv_results[i]['test-rmse-mean'], label=str(cv_results[i]['test-rmse-mean'].name)+' parameters: '+str(value))
        plt.plot(cv_results[i].index.values, cv_results[i]['train-rmse-mean'], label=str(cv_results[i]['train-rmse-mean'].name)+' parameters: '+str(value))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('RMSE')
    plt.ylim(2200,2600)
    plt.title("XGB parameter gridsearch - n_estimators vs RMSE")
    plt.show()
#%%
xgb_gridserach_results_plot_(cv_results, gridsearch_params)
#%%
# plot performance for interesting scores x due to scaling:
i = 0
plt.figure(figsize=(16,12))
plt.plot(cv_results[i].index.values, cv_results[i]['test-rmse-mean'], label=str(cv_results[i]['test-rmse-mean'].name)+' parameters: '+str(gridsearch_params[i]))
plt.plot(cv_results[i].index.values, cv_results[i]['train-rmse-mean'], label=str(cv_results[i]['train-rmse-mean'].name)+' parameters: '+str(gridsearch_params[i]))
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.title("XGB parameters...  n_estimators vs RMSE")
plt.legend()
plt.show()
#%%
params['max_depth'] = 3
params['min_child_weight'] = 26
#%%
# tuning subsample and colsample_bytree
gridsearch_params = [
    (subsample, colsample)
#    for subsample in [i/10. for i in range(7,11)]
#    for colsample in [i/10. for i in range(7,11)]
    for subsample in [1]
    for colsample in [i/10. for i in range(4,11)]
    # maybe also tune colsample_bylevel in [i/10. for i in range(4,11)]
]

#subsample and colsample by tree
#cv_results = xgb_native_gridsearch(dtrain, params, num_boost_round = 100000, folds=cv_split, gridsearch_params = gridsearch_params, parameter_pairs = ("subsample", "colsample_bytree"), verbose_eval = 400, early_stopping_rounds = 100)
#colsample_bylevel
cv_results = xgb_native_gridsearch(dtrain, params, num_boost_round = 100000, folds=cv_split, gridsearch_params = gridsearch_params, parameter_pairs = ("subsample", "colsample_bylevel"), verbose_eval = 400, early_stopping_rounds = 100)

#Best params:  subsample=1.0, colsample_bytree=0.8, colsample_bylevel=1.0 RMSE: 2487.3324708

#Best params for the interaction dataset with k = 39 features --> xgb_thresh:
#Best params:  subsample=1.0, colsample_bytree=0.9, RMSE: 2497.0865967
#%%
#update dict # some values for xgb_thresh
#params['subsample'] = 1.
#params['colsample_bytree'] = 0.9


# params for original dataset, tuning lambda now
params['subsample'] = 1.
params['colsample_bytree'] = 0.8
#%%
gridsearch_params = [
        (reg_alpha, reg_lambda) 
        for reg_alpha in [0]#, 0.01, 0.1, 0.3]
        for reg_lambda in [1., 3., 10., 30., 50, 100, 150, 200, 300] # 150,200,300 are new    
        ]

cv_results = xgb_native_gridsearch(dtrain, params, num_boost_round = 1000000, folds=cv_split, gridsearch_params = gridsearch_params, parameter_pairs = ("alpha","lambda"), verbose_eval = 400, early_stopping_rounds = 100)


#\Best params:  alpha=0.0, lambda=50.0, RMSE: 2470.193457
#total runtime is -5:36:21
#Best params for the interaction dataset with k = 39 features --> xgb_thresh:
# best value so far: alpha 0, lambda 100 --> RMSE~2492, bigger than with original set. 
# will try with better feature engineering again

#%%
params['alpha'] = 0.0
params['lambda'] = 30.
#%%
gridsearch_params = [
        (eta, num_boost_round) 
        for eta in [.3, 2.5, .2, 1.5, .1, .05]
        for num_boost_round in [100000]    
        ]

cv_results = xgb_native_gridsearch(dtrain, params, num_boost_round = 100000, folds=cv_split, gridsearch_params = gridsearch_params, parameter_pairs = ("eta","num_boost_round"), verbose_eval = 400, early_stopping_rounds = 100)
# Best params:  eta=0.1, num_boost_round=100000, RMSE: 2468.7594726
#%%
#cv_results[0]
#%%
# check for best tradeoff between rmse and time
params['eta'] = 0.2
#%%
# test effect of gamma 
# colsample_bylevel can also be looked at for fine tuning
gridsearch_params = [
        (eta, gamma) 
        for eta in [0.1]
        for gamma in [10,30,90,150]#[0,1,10,30,100] # high gamma only needed for very high depth trees to control overfitting over all other parameters. 
        # Should tune gamma last    
        # We didn't get any different RMSE for gamma up to 5 though. Do we need like 20 or 50? Can try effects of this on train and test cv score
        ]

cv_results = xgb_native_gridsearch(dtrain, params, num_boost_round=100000, folds=cv_split, gridsearch_params = gridsearch_params, parameter_pairs = ("eta","gamma"), verbose_eval=400, early_stopping_rounds = 100)
# Best params:  eta=0.1, gamma=0, RMSE: 2468.7594726
"""
#%%
"""

params = {
    # Parameters for rour interaction dataset 
    #with feature importance picked 
    #top features based on cv
    'max_depth':4,
    'min_child_weight': 13,
    'eta':.2,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'nthread' : 8,
    # Other parameters
    'objective':'reg:linear',
    'alpha': 0.0,
    'lambda': 30.,
    'gamma': 0.,
    'seed' : 42,
}
"""
params = {
        # final param list
    # Parameters for our full dataset without interaction terms
    'max_depth':4,
    'min_child_weight': 13,
    'eta':.2,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'nthread' : 8,
    'objective':'reg:linear',
    'alpha': 0.0,
    'lambda': 30.,
    'gamma': 0.,
    'seed' : 42,
}

#%%
#Let’s train a model with it and see how well it does on our val set!
#get number of trees first
num_boost_round = 100000
model = xgboost.train(
    params,
    dtrain,
    verbose_eval = 400,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=800
)

print("optimal number of estimators is: ", int(model.best_ntree_limit/(1-0.1))) # eval set is 0.1
print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
xgboost.plot_importance(model)
# and train the model with final parameters and early stopping
# train with best iteration or best_ntree_limit

#optimal number of estimators is:  12413
#Best RMSE: 2472.72 in 11172 rounds

# second tuning round (gamma = 90)
#optimal number of estimators is:  16668
#Best RMSE: 2477.92 in 15002 rounds

# for eta 0.2 and lambda 50
#optimal number of estimators is:  11210
#Best RMSE: 2475.09 in 10089 rounds

# first tuning parameters with early stopping 800:
#optimal number of estimators is:  14216
#Best RMSE: 2471.92 in 12795 rounds

# second tuning round (gamma = 0,  early stopping 1700)
#optimal number of estimators is:  33055
#Best RMSE: 2469.21 in 29750 rounds

# second tuning round (gamma = 0,  early stopping 850)
#optimal number of estimators is:  28738
#Best RMSE: 2469.83 in 25865 rounds

# second tuning round (gamma = 0,  early stopping 400)
#optimal number of estimators is:  19683
#Best RMSE: 2474.91 in 17715 rounds
#%%
# test .best_ntree_limit
print("best model prediction RMSE: ",np.sqrt(mean_squared_error(y_test,model.predict(dtest, ntree_limit=model.best_ntree_limit))))
# RMSE: 2472.72
# can use this to predict with best trees

# rmse for martial status removed with early stopping - compared with the model above (2472.72 RMSE) with full predictors
# Best RMSE: 2476.55 in 7438 rounds ---> worse than with full predictors
#%%
# averaging / ensembling the xgb predictions results in a little better RMSE for us
"""
# train with full train set
data = data1
predictors = data1_code
target = "Purchase"
num_boost_round =  14216 

dtrain = xgboost.DMatrix(data[predictors].values, label=data[target].values, nthread = 8)
dtest = xgboost.DMatrix(data_val[predictors].values, missing = np.nan, nthread = 8)

best_model = xgboost.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
)

# predict on test set
test_preds  = best_model.predict(dtest) # can use ensembling to average predictions as well and see if it has a positive effect on RMSE
# we had RMSE of 2474 on this with 12413 trees
# RMSE of 2472 with 11172 trees
# RMSE of 2472 with 28738 trees and second tuning
"""
#%%
plot_learning_curves_XGB_reg(params, data, predictors, target, suptitle = "learning curve tests", title="XGB tuned "+str(model.__class__.__name__)+" CV scores", xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.9, test_size = 0.1,  scale = False, center = False, split = "KFold", random_state = 42)
# still a good bit of variance, but so much better than before
#%%
# After model tuning we can ensemble the models and average the predictions

num_boost_round = 14216 
#29750 early stop 1700 and new params
# 33055 with early stopping 1700 and new params 
#14216 for base set with first parameter tunes and 800 early stopping rounds  
#15002 # 16668 - RMSE 2475  
#12413 - RMSE 2467 
data = data1
predictors = data1_code
dtrain = xgboost.DMatrix(data[predictors].values, label=data[target].values, nthread = 8)
dtest = xgboost.DMatrix(data_val[predictors].values, missing = -1, nthread = 8)

seeds = [42, 2343, 3543, 4556, 5696]
test_preds = np.zeros((data_val.shape[0], len(seeds)))

for run in range(len(seeds)):
    print("\rXGB run {} / {} ".format(run+1, len(seeds)))
    params["seed"] = seeds[run]
    # train xgboost
    reg = xgboost.train(params, dtrain, num_boost_round)
    test_preds[:,run] = reg.predict(dtest)

test_preds = np.mean(test_preds, axis=1)

# use this for submission as well
# new highscore with 2467.1 RMSE # num_boost_round = 12413
# 2467.5 RMSE # num_boost_round = 11172

# 16668 rounds with second tuning of parameters
# 2475 RMSE...

    #14216 with early stopping 800
         # RMSE 2467... best result so far #### #####

#29750 with early stopping 1700
 #RMSE 2468...
#%%
# =============================================================================
# #submission!
# =============================================================================
submit = pd.DataFrame({"User_ID": test.User_ID, "Product_ID": test.Product_ID, "Purchase": test_preds})
submit = submit[["User_ID", "Product_ID", "Purchase"]] # need to change order of the columns in the DF!
#%%
# we should only predict positive purchase values 
# some values are negative - makes no sense to predict negative values of course 
# Set lowest Purchase value for predicted negative values
print(np.percentile(data1["Purchase"], 0))
lowest_purchase_amount = np.percentile(data1["Purchase"], 0)
#%%
# one solution to change neg. values to min value of train in submit DF.
submit["Purchase"] = np.where(submit["Purchase"]<0,  lowest_purchase_amount, submit["Purchase"])
# with submit.loc or iloc second one?
#submit.loc[submit["Purchase"]< 0,  "Purchase"] = lowest_purchase_amount
#%%
#submit.to_csv('E:/datasets/black friday/xgb_data1_code_nomax_1000trees_notuning_label_encoding_-1test.csv', index=False) # 2578.75253539474
#submit.to_csv('E:/datasets/black friday/xgb_data1_code_maxpurchase_1000trees_notuning_label_encoding_-1test.csv', index=False) # 2577.10638661009
#submit.to_csv('E:/datasets/black friday/xgb_pred_final__maxpurchase_1000trees_notuning_label_encoding_-1test.csv', index=False) # 2695.13244006872
#submit.to_csv('K:/datasets/black friday/xgb_pred_test_nomax_1000trees_notuning_label_encoding_-1test.csv', index=False) # 2596.91879629019
#submit.to_csv('K:/datasets/black friday/xgb_rfe_preds_nomax_1000trees_notuning_label_encoding_-1test.csv', index=False) # 2578.17084194027
#submit.to_csv('K:/datasets/black friday/xgb_data1_code_nomax_1500trees_notuning_label_encoding_-1test.csv', index=False) # 2543.588488356
#submit.to_csv('K:/datasets/black friday/xgb__data1_code__nomax__6628trees__tuning_max_depth5___min_child_weight15__label_-1test.csv', index=False) # 2476.16027299912
#submit.to_csv('K:/datasets/black friday/xgb__data1_code__nomax__8287trees__tuning_max_depth5___min_child_weight14__label_-1test.csv', index=False) # 2475.72339063536
#submit.to_csv('K:/datasets/black friday/xgb__data1_code__nomax__7630trees__tuning_max_depth5___min_child_weight14__colsample085__subsample095__label_-1test.csv', index=False) # 2471.06341097805
#submit.to_csv('K:/datasets/black friday/xgb__data1_code__nomax__9537trees__tuning_max_depth5___min_child_weight14__colsample085__subsample095__label_-1test.csv', index=False) # 2470.66726779014
#submit.to_csv('K:/datasets/black friday/xgb__data1_code__max_removed__8861trees__tuning_max_depth5___min_child_weight14__colsample085__subsample095__label_-1test.csv', index=False) # 2470.5494881216
#submit.to_csv('K:/datasets/black friday/xgb__data1_code__max_removed__7089trees__tuning_max_depth5___min_child_weight14__colsample085__subsample095__label_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_nativeapi_data1_code__nomax__12413trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__label_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_nativeapi_data1_code__nomax_5ensembleaveraged_12413trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__label_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_nativeapi_data1_code__nomax__11172trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__label_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_nativeapi_data1_code__nomax_5ensembleaveraged_11172trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__label_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_nativeapi_data1_code__nomax_5ensembleaveraged_12413trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__labelsrevamp.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_nativeapi_data1_code__nomax_5ensembleaveraged_16668trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__labelsrevamp_second_tuning.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_missing_-1_data1_code__nomax_5ensembleaveraged_14216trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__labelsrevamp_second_tuning.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_missing_-1_data1_code__nomax_5ensembleaveraged_33055trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__labelsrevamp_second_tuning.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_missing_-1_data1_code__nomax_5ensembleaveraged_28888trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__labelsrevamp_second_tuning.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_missing_-1_data1_code__nomax_5solo28738trees__tuning_max_depth4___min_child_weight13__colsample08__subsample1__labelsrevamp_second_tuning.csv', index=False)
submit.to_csv('K:/datasets/black friday/xgb_missing_-1_data1_code__nomax_5ensembled_FINAL-SUB_14216trees.csv', index=False)

#%%
"""
# =============================================================================
# #Next we try ensembling. Might have to tune XGB all over again for new meta features.
# #First results showed now improvement over tuned solo XGB.
# 
# ### Gridsearch for our Meta Stacking Ensemble ###
# #Hyperparameter tune with GridSeachCV
# =============================================================================
data = data1
target = "Purchase"
random_state = 42
predictors = data1_code

grid_n_estimator = [100,300,400]
grid_ratio = [0.03, .1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_coeffs = [0, 0.1, 1, 10]
grid_gamma = [0.001, 0.01, 0.1, 1, "auto"]
#grid_max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]
grid_max_depth = [3, 10, None]
grid_min_leaf_samples = [1, 10, 30]
grid_max_leaf_nodes = [5, 10, 30]
#grid_min_samples = [5, 10, .03, .05, .10]
grid_min_samples = [2,10]
grid_degree = [2, 3, 4]
grid_criterion = ["gini", "entropy"]
grid_bool = [True, False]
grid_seed = [42]
grid_regular = [0.00001, 0.0001, 0.001, 0.01, 0.1]
grid_features = [int(len(predictors)/2)+1, len(predictors)]

grid_param_level2 = [
      #[{
      #SVR
      #"kernel": ["linear", "poly", "rbf", "sigmoid"],
      #"C": [0.01, 0.1, 1, 10, 100, 1000], 
      #"gamma": ["auto"],
      #"degree": grid_degree,
      #"tol" : grid_regular,
      #"coef0": grid_coeffs,
      #"epsilon": [0.01, 0.1, 0.3],
      #"random_state": grid_seed,
      #"verbose" :  [True]
      #}],  
        
       # [{ 
      #NuSVC
      #"nu" : [0.1, 0.5, 0.8, 0.9],
      #"kernel": ["linear", "poly", "rbf", "sigmoid"],
     # "coef0": grid_coeffs,
      #"gamma": grid_gamma,
      #"tol" : grid_regular,
      #"decision_function_shape": ["ovo", "ovr"], 
      #"degree" : grid_degree,
      #"probability": [True],
      #"random_state": grid_seed
      #}], 
      
       [{ 
      #TheilSenRegressor # {'max_iter': 2000, 'max_subpopulation': 10000, 'n_subsamples': None, 'random_state': 42, 'tol': 0.0001, 'verbose': True}
      "max_subpopulation": [1000,5000], 
      "n_subsamples": [None],
      "max_iter": [100,300,1000],
      "tol": [0.0001],
      "random_state": grid_seed,
      "verbose": [True],
      }], 
    
        ]


grid_param_level1 = [
        
      [{
      #LinearRegression
      "fit_intercept": grid_bool, 
      }],
        

    
     [{ 
      #ElasticNet
      "alpha": [.1, .5, 1.0],
      "l1_ratio": [0, 0.5, 1],
      "max_iter": [2000],
      "tol": [0.0001, 0.001, 0.01],
      "selection": ["random"],
      "fit_intercept": grid_bool, 
      "warm_start": [True],
      "random_state": grid_seed
      }],  
    
     [{ 
      #RANSAC
      "min_samples": [1000, 13],
      "max_trials": [2000],
      "fit_intercept": grid_bool, 
      "random_state": grid_seed
      }],  
        ]

grid_param_level0 =[
        
    # check decision tree as well
    # +adaboost 
    
     #   [{
      #ExtraTreesRegressor # maybe we can converge faster with extremelyrandomforests?
     # "n_estimators": grid_n_estimator, #def = 10
    #  "criterion": grid_criterion, #def = "gini"
    #  "max_depth": grid_max_depth, #def = None
      #"oob_score": [True], #def = False
    #  "min_samples_split" : grid_max_depth,
    #  "min_samples_leaf" : grid_max_depth,
      #"bootstrap": [True],
    #  "random_state": grid_seed,
  #    }],
      
        [{
      #RandomForestRegressor #{'max_depth': None, 'max_features': 6, 'min_samples_split': 10, 'n_estimators': 500, 'oob_score': True, 'random_state': 42}
      "max_features" : grid_features, # 11 was good 
      "n_estimators": grid_n_estimator, #def = 10  #300 was good
      "max_depth": grid_max_depth, #def = None # 10 was good  
      # max_depth should only be tuned without setting values for min_samples_leaf and max_leaf_nodes - they cancel out the effect of setting max_depth
      "oob_score": [True], #def = False
      "min_samples_split" : grid_min_samples, # 10 was good
      "min_samples_leaf" : grid_min_leaf_samples,
      "max_leaf_nodes": grid_max_leaf_nodes, 
      "random_state": grid_seed
      }],
  
    [{
      #KNeighborsRegressor
      "n_neighbors": [x for x in range(11,12)], #KNeighborsRegressor {'n_neighbors': 11 - weights: distance with kd_tree best ( 14276970...)
      "weights": ["uniform", "distance"], 
      "algorithm": ["ball_tree", "kd_tree", "brute"] # some take a very long time. we'll go with what we have now
      }],   
   ]
#%%
MLA1 = [
        #("etc", ensemble.ExtraTreesRegressor()),
        ("rfr", ensemble.RandomForestRegressor()),
        ("knn", neighbors.KNeighborsRegressor()),
    ]  

MLA2 = [
        #GLM
       ("lr", linear_model.LinearRegression()),
       
       ("elnet", linear_model.ElasticNet()),
       ("RANSAC", linear_model.RANSACRegressor()),
       ]    

MLA3 = [
        # GLM2
       ("TheilSen", linear_model.TheilSenRegressor()),
        ]   
#%%        
import time
def gridsearch_params_mla(MLA, param_grid, data, predictors, target, cv=10, scoring = "accuracy"):
    Grid_compare_cols = ["clf/reg", "best params", "runtime[s]"]
    Grid_compare = pd.DataFrame(columns=Grid_compare_cols )
    row_index = 0
   
    start_total = time.perf_counter()
    for clf, param in zip (MLA, param_grid):
        #MLA is a list of tuples, index 0 is the name and index 1 is the algorithm
        #grid_param is a list of param_grids for the gridsearch for each estimator
        # do param search
        start = time.perf_counter()
        print("started with ", clf[0])
        best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv, iid=False, scoring = scoring, verbose = 10, return_train_score = True)
        best_search.fit(data[predictors], data[target])
        
        #get the run time for that clf
        run = time.perf_counter() - start
        
        #get best params and set them to the clf in the MLA list
        best_param = best_search.best_params_
        print("The best parameter for {} is {} with a runtime of {:.2f} seconds.".format(clf[1].__class__.__name__, best_param, run))
        clf[1].set_params(**best_param)
        
        # store results
        Grid_compare.loc[row_index, "clf/reg"] = str(clf[1].__class__.__name__)
        Grid_compare.loc[row_index, "best params"] = str(best_param)
        Grid_compare.loc[row_index, "runtime[s]"] = "{:.2f} ".format(run)
        row_index+=1
        
    run_total = time.perf_counter() - start_total
    print("Total optimization time was {:.2f} minutes.".format(run_total/60))
    print("-"*10)
    
    # get to the parameters to copy them in
    for i in range(0,Grid_compare.shape[0]):
        print(Grid_compare.iloc[i,0])
        print(Grid_compare.iloc[i,1])
#%%  
# need to scale the data for some MLAs
dataset = data1
predictors = data1_code
predictors_scale = data1_code # all feature names
target = "Purchase" # response name

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaled = scaler.fit(dataset[predictors_scale])
scaled_data = scaled.transform(dataset[predictors_scale])
scaled_data = pd.DataFrame(scaled_data, columns = predictors_scale )
scaled_data = pd.concat([scaled_data, dataset[target]], axis=1)
#%%
# needed memory space --> cleaning
import gc
del_list= [data1,data,data_val,dataset,test,train, df, df_interaction]
for item in del_list:
    del item
gc.collect()
#%%
# we only tune knn and rfr because I realized that linear models suffer a lot from the approach of not creating dummy variables
# we will get CV results for dummy variables and maybe also interaction terms with removed correlation (center data first)
# CV results for dummy variables themselves are already much improved compared to just labelencoded variables - about 1800 in RMSE decrease!
gridsearch_params_mla(MLA1, grid_param_level0, scaled_data, predictors, target, cv = cv_split, scoring = "neg_mean_squared_error")
#%%
# only use this with dummy variables, no sense otherwise
gridsearch_params_mla(MLA2, grid_param_level1, scaled_data, predictors, target, cv = cv_split, scoring = "neg_mean_squared_error")
#%%
# only use this with dummy variables, no sense otherwise
gridsearch_params_mla(MLA3, grid_param_level2, scaled_data, predictors, target, cv = cv_split, scoring = "neg_mean_squared_error")

#%%
### STACKED Meta Ensembling  ###

# need to use gridsearch on most models to tune parameters #
def Stacking(model, X_train, X_test, y_train, n_fold, scale = False, center = False):
    #define a scaling function
    def scaling(X_train, X_val, scale, center):
        do_scale = 0
        
        if (scale == True) & (center == False):
            with_std = True
            with_mean = False
            do_scale = 1
        elif (scale == False) & (center == True):
            with_std = False
            with_mean = True
            do_scale = 1
        elif (scale == True) & (center == True):
            with_std = True
            with_mean = True
            do_scale = 1
            
        if do_scale == 1:
            
            #scale data now! - fit on train, transform on both
            scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
            scaled = scaler.fit(X_train)
            X_train = scaled.transform(X_train)
            X_train = pd.DataFrame(X_train)
            
            # apply scaling to test set
            X_val = scaled.transform(X_val)
            X_val = pd.DataFrame(X_val)
        return X_train, X_val
    
    # split the data into training and validation sets
    ntest = X_test.shape[0]
    ntrain = X_train.shape[0]
    
    folds=KFold(n_splits=n_fold, random_state=42, shuffle = True)

    train_pred = np.zeros((ntrain,))    
    test_pred = np.zeros((ntest,)) 
    test_pred_skf = np.empty((n_fold, ntest))
    
    # run the models on k folds of the training set and predict on train and validation set
    for i, (train_indices, val_indices) in enumerate(folds.split(X_train,y_train)):
        x_tr, x_val = X_train.iloc[train_indices], X_train.iloc[val_indices]
        y_tr = y_train.iloc[train_indices]#, y_train.iloc[val_indices] 
        # we don't use y_val since we don't want the model to see that in fit anyway
        # we check our predictions on the test set only at the end 
        
        
        # scale the data for the algorithms that need it
        if (scale == True) or (center == True):
            print("Scaling for fold k = ",i+1)
            _, X_test_scaled = scaling(x_tr, X_test, scale, center)
            x_tr, x_val = scaling(x_tr, x_val, scale, center)
        else:
            print("Not scaling. Fold k = ",i+1)
            X_test_scaled = X_test
        
        #fit the models to every combined train subset and predict on x_val 
        model.fit(X=x_tr, y=y_tr)
        train_pred[val_indices] = model.predict(x_val)
        test_pred_skf[i, :] = model.predict(X_test_scaled)
        
    #test_pred[:] = sp.stats.mode(test_pred_skf)[0] # should we rather use max voting and return the mode - only for clfs
    test_pred[:] = test_pred_skf.mean(axis=0) # for reg we use mean 
    return test_pred, train_pred

def stacking_modelfit(model, X_train, X_test, y_train, folds, scale = False, center = False):
    test_pred, train_pred = Stacking(model, X_train, X_test, y_train, folds, scale, center)
    train_pred = pd.DataFrame([train_pred]).T
    test_pred = pd.DataFrame([test_pred]).T
    print("stacking modelfit is done")
    return train_pred, test_pred

# scale data now! - fit on train, transform on both. 
# Original data needs to be scaled already so we can apply the same algorithms that work with scaled data on actually scaled data  
# In a better workflow we might return the scaler (on x_tr) from the functions and use it on our data outside 
# Or we do this in a better pipelined / CV version on the whole train set. 
#Would take so much longer however and we don't do this here.
# could put this in a cv loop and use the whole train set. 
#Could then average the fold predictions together and use the averages as meta features
#and predictions on test set

random_state = 42
data = data1

predictors = data1_code
target = "Purchase"
# Hold back part of train set and split into train and val set
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=0.1, shuffle = True, random_state=random_state)
    

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaled = scaler.fit(X_train)
X_train = scaled.transform(X_train)
X_train = pd.DataFrame(X_train, columns = predictors)
# apply scaling to test set
X_test = scaled.transform(X_test)
X_test = pd.DataFrame(X_test, columns = predictors)
     
  
# level 0 
model1 =  ensemble.RandomForestRegressor(max_depth = None, max_features = 6, min_samples_split = 10, n_estimators = 500, oob_score = True, random_state = 42) 
train_pred1, test_pred1 = stacking_modelfit(model1, X_train, X_test, y_train, 5)

model2 = neighbors.KNeighborsRegressor(n_neighbors = 11, weights = "distance")  
train_pred2, test_pred2 = stacking_modelfit(model2, X_train, X_test, y_train, 5)

# work the results in a dataframe together with original data
df1 = pd.concat([train_pred1, train_pred2],  axis=1)
df_test1 = pd.concat([test_pred1, test_pred2],  axis=1)

df1.columns = ["RFR", "KNN"]
df_test1.columns = ["RFR", "KNN"]

df_train = X_train.reset_index(drop=True) #ex: X_train.copy().reset_index(drop=True)
df_train = pd.concat([df_train, df1], axis = 1) # concatenate original data with new "predict features"

df_test = X_test.reset_index(drop=True)
df_test = pd.concat([df_test, df_test1], axis = 1)

# correlations of models
correlation_heatmap(df_train)


#%%
# =============================================================================
#  level 1 = final level in this case
# =============================================================================
# predict on new features as well in the final model
# tune the model again before predictions...

# define DMatrices for tuning
dtrain = xgboost.DMatrix(df_train.values, label=y_train.values, feature_names =  df_train.columns , nthread = 8)
dtest = xgboost.DMatrix(df_test.values, label=y_test.values, feature_names = df_test.columns, nthread = 8)

#%%
# =============================================================================
# param list for tuning the stacked model
# =============================================================================
params = {
    # Parameters that we are going to tune.
    'max_depth':4,
    'min_child_weight': 13,
    'eta':.2,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'nthread' : 8,
    'objective':'reg:linear',
    'alpha': 0.0,
    'lambda': 30.,
    'gamma': 0.,
    'seed' : 42,
}
#%%
num_boost_round = 100000
model = xgboost.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=600
)
print("optimal number of estimators is: ", int(model.best_ntree_limit/(1-0.1))) # eval set is 0.1
print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
#optimal number of estimators is:  6555
#Best RMSE: 2517.15 in 5900 rounds
#%%
print("best model prediction RMSE:  %.5g" % np.sqrt(mean_squared_error(y_test,model.predict(dtest, ntree_limit=model.best_ntree_limit))))
plt.figure(figsize=(20,15))
xgboost.plot_importance(model, ax=plt.gca())
#%%

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in [2,3,4]
    for min_child_weight in [3,5,13]
]
#%%
cv_results = xgb_native_gridsearch(dtrain=dtrain, params=params, num_boost_round = 100000, folds=cv_split, gridsearch_params = gridsearch_params, parameter_pairs = ("max_depth", "min_child_weight"), verbose_eval = 300, early_stopping_rounds = 600)

#Best params:  max_depth=4, min_child_weight=5, RMSE: 2531.2381837999997
#Total runtime is 3:48:9
# ok we could try to tune much more for our stacked model... but this would result in even more time invested for possibly marginal results
# we are not even getting below 2500 RMSE right now. Might try adjusting sub and colsample plus lambda and gamma...
#%%
# print results
means = [cv_results[x]['test-rmse-mean'] for x in range(len(cv_results))]
stds =  [cv_results[x]['test-rmse-std'] for x in range(len(cv_results))]
params = gridsearch_params
for mean, stdev, param in zip(means, stds, params):
    print("min test mean rmse: {:.2f}+{:.2f}std with params max_depth={} and min_child_weight={}".format(mean.min(),
          stdev[mean.idxmin()], param[0], param[1]))
#%%
# plot results
plt.figure(figsize=(16,12))
for i, value in enumerate(gridsearch_params):
    plt.plot(cv_results[i].index.values, cv_results[i]['test-rmse-mean'], label=str(cv_results[i]['test-rmse-mean'].name)+' parameters: '+str(value))
    plt.plot(cv_results[i].index.values, cv_results[i]['train-rmse-mean'], label=str(cv_results[i]['train-rmse-mean'].name)+' parameters: '+str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.ylim(2200,2650)
plt.title("XGB parameter gridsearch - n_estimators vs RMSE")
plt.show()


#%%

num_boost_round = 100000
model = xgboost.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=100
)

print("optimal number of estimators is: ", int(model.best_ntree_limit/(1-0.1))) # eval set is 0.1
print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
xgboost.plot_importance(model)

#predictions of tuned stacked-model
print("best model prediction RMSE:  %.5g" % np.sqrt(mean_squared_error(y_test,model.predict(dtest, ntree_limit=model.best_ntree_limit))))
# RMSE of blended model without tuning the individual models is 2552.6, worse than single XGB yet.
# RMSE with tuning: Blended model RMSE:  2527.9 

# Base model tuned has less RMSE


#%%
# plot learning curve for stacked model
plot_learning_curves_XGB_reg(params, df_train, df_train.columns, target, suptitle = "learning curve tests", title="XGB tuned "+str(model.__class__.__name__)+" CV scores", xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.9, test_size = 0.1,  scale = False, center = False, split = "KFold", random_state = 42)
"""


#%%

