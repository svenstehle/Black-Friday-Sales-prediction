# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy as sp
from scipy.stats import pearsonr

# some imports
#from sklearn_pandas import DataFrameMapper
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score, GridSearchCV, KFold, ShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, NMF 
    # Non-Negative Matrix Factorization (NMF)
    # Find two non-negative matrices (W, H) whose product approximates the non- negative matrix X. 
    # This factorization can be used for example for dimensionality reduction, source separation or topic extraction.
from sklearn.svm import SVR
from sklearn import tree, linear_model, neighbors, svm, ensemble
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet, RANSACRegressor
import xgboost

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Imputer, MinMaxScaler
from sklearn import model_selection, feature_selection, metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import chi2, f_regression, SelectKBest # feature extraction with Statistical Selection

from xgboost.sklearn import XGBRegressor
import matplotlib.pylab as pyl
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
#pyl.rcParams["figure.figsize"] = 12,8
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
# User_ID and all the product categories are included in the test data.
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
print(pd.DataFrame(train.sort_values("User_ID")["User_ID"].unique()).count())
print(pd.DataFrame(test.sort_values("User_ID")["User_ID"].unique()).count())
# we have the same amount of unique values in the test and train set for the User_ID
# can also use .nunique() instead of unique and count together.
#%%
print(np.all(True in (train.sort_values("User_ID")["User_ID"].unique() == test.sort_values("User_ID")["User_ID"].unique())))
# variables User_ID are identical in train and test set!
#%%
# Do we also have identical Product_ID elements in the test set?
print(pd.DataFrame(train.sort_values("Product_ID")["Product_ID"].unique()).count())
print(pd.DataFrame(test.sort_values("Product_ID")["Product_ID"].unique()).count())
# We have a different set of Product_ID elements in the test and train data set!
#%%
print(len(set(train["Product_ID"]) - set(test["Product_ID"])) )
product_ids_train_only = (set(train["Product_ID"]) - set(test["Product_ID"])) # 186 are absent in test
print(len(set(test["Product_ID"]) - set(train["Product_ID"]) )) 
new_product_ids_test = (set(test["Product_ID"]) - set(train["Product_ID"]) ) # 46 are new in test

# need to take care about unknown IDs and possible labelencoding in test.
#%%
test["Product_ID"].isin(new_product_ids_test)
#%%
# could set those to -1, for example
test.ix[test["Product_ID"].isin(new_product_ids_test) , "Product_ID"]
#%%
# could label encode the others with labels from train
test.ix[~test["Product_ID"].isin(new_product_ids_test) , "Product_ID"]
#%%
#sns.countplot(y="Purchase", hue="Gender", data=train)
#plt.show()
#%%
# we can feature engineer maybe interesting combinations - combine marital status and age bins, age, job or gender and city category etc.
# put this on ice for now and maybe improve in a future version <---
# if we use boosted decision trees, will they pick out the relationships we could create by engineering the features manually?
data1.info()

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
# Gender makes a difference. Men spend more. Are those men married? Let's take a look at that later.
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
print(data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).max().sort_values(Target[0], ascending = False))
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
print(data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).sum().sort_values(Target[0], ascending = False))

## interesting, in some cases we have quite a big turnover per product. What is the total turnover?
print("-"*10)
print("total turnover is: \n", data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).sum()["Purchase"].sum()/1000000, " million $")
## so we have quite a big company with over 5 billion $ turnover - assuming that the value of Purchase is given in USD. We have no information on that.
#%%
# our purchase sum for any unique product ID
data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).sum()["Purchase"]
#%%
# more visualization on products and purchases
turnover = data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).sum()["Purchase"]
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
plt.xlabel("Purchase sum per product ID [$]")
plt.ylabel("log # of purchases")
plt.show()

# it looks more like a histogram of normal distribution. What's the value in that for us here? Would love advice or insights.
#%%
#our counts of the purchases for any unique product ID
data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).count()["Purchase"]
#%%
# more visualization on products and purchases - count of purchases for unique product ID
products_num_purchases = data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).count()["Purchase"]
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
plt.xlabel("# of purchases per product ID")
plt.ylabel("log # of products")
plt.legend()
plt.show()

# in what way does log transformation help us with interpretability?
# to my eye, it looks like the untransformed graph gives us more 
## (easily accessible?) information
# Advice & comments welcome!

## We can conclude that most purchases have a small turnover and less purchases have a large turnover
## Also, most products have small amount of purchases, less products have a large amount of purchases



#%%
# how many unique products do we have in train (we know it's more than in test)? Double Check
data1[["Product_ID", Target[0]]].groupby("Product_ID", as_index = False).sum().sort_values(Target[0], ascending = False)["Product_ID"].nunique()
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
##### experimental #####
# How many customers did make more than one purchase? Counts of Purchases for customer
cust = data1[["User_ID", Target[0]]].groupby("User_ID", as_index = False).count()["Purchase"]

plt.figure(figsize=(12,8))
plt.hist(x= cust, bins=45)
plt.title("Purchase per customer Histogram")
plt.xlabel("Purchase count")
plt.ylabel("# of customers with that count of purchases")
plt.show()
# similar distribution as for the Product_ID count.
#%%
sns.countplot(y="Purchase", hue="User_ID", data=train)
plt.show()
# We could split the distribution at 200 purchases - more than 200 equals 1, less than 200 equals 0
# Then set the Purchase count for each User_ID in a separate column
#%%
# We have many unique Product_IDs. We LabelEncode these and maybe create dummy variables. Test with CV later 
#%%
data1.sample(5)
#%%
data1.info()
#%%

### use of MAP AND REPLACE for preprocessing ######
# need to assign to new or existing column for it to work inplace #
print(data1["Gender"].map({"M": 0, "F": 1}))
print(data1["City_Category"].replace({"A": 1, "B": 2, "C": 3}))
print(data1.sample(5))
#%%
# label encoding for different algorithms
label = LabelEncoder()  #encodes objects to categorical integers 
for dataset in data_cleaner:
    dataset["User_ID_code"] = label.fit_transform(dataset["User_ID"])  
    dataset["Gender_code"] = label.fit_transform(dataset["Gender"])
    dataset["Age_code"] = label.fit_transform(dataset["Age"])
    dataset["City_Category_code"] = label.fit_transform(dataset["City_Category"])
    dataset["Stay_In_Current_City_Years_code"] = label.fit_transform(dataset["Stay_In_Current_City_Years"])
    
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
data_val.loc[data_val["Product_ID"].isin(new_product_ids_test) , "Product_ID_code"] = -1 # the correct label encoding in test gave us over 200 RMSE decrease!
#%%
# filling NaN values with 0. 0 was not present in categories before the fill
# change dtype to int, since we only have .0 values in there
for dataset in data_cleaner:
    
    dataset["Product_Category_2"].fillna(0, inplace = True) # compare RMSE without category 2 and 3 later and without imputing!
    dataset["Product_Category_3"].fillna(0, inplace = True)
    dataset["Product_Category_2"] = pd.DataFrame(dataset["Product_Category_2"], dtype = "int64")
    dataset["Product_Category_3"]  = pd.DataFrame(dataset["Product_Category_3"], dtype = "int64")

#%%
data1_x = ["User_ID", "Product_ID", "Gender", "Age", "Occupation", 
           "City_Category", "Stay_In_Current_City_Years", "Marital_Status", 
           "Product_Category_1", "Product_Category_2", "Product_Category_3"]
ABC = "A B C D E F G H I J K L M".split()

#%%
# for different tests
data1_x_purchase = ["User_ID", "Product_ID", "Gender", "Age", "Occupation", 
           "City_Category", "Stay_In_Current_City_Years", "Marital_Status", 
           "Product_Category_1", "Product_Category_2", "Product_Category_3", "Purchase"]

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
sns.pairplot(data1[data1_code_purchase])

## Too many datapoints, now new insights for me. Log transform helpful here? Not on objects though
#%%
for x, letter in zip(data1_x,ABC):
    print(letter,") Purchase Grouping by: ", x)
    print(data1[[x, Target[0]]].groupby(x, as_index = False).mean().sort_values(Target[0], ascending = False))
    print("-"*10)

# Going through this one at a time, we can see that:
# A) User_ID is something we need for our predictions later, 
    # since we are supposed to predict future Purchase value on past Purchase value for each bought Product_ID for distinct customers.
    # second thoughts: we don't need the User_ID in general. Does ist reduce error though? It could be something that introduces noise.
# B) First thoughts: Product_ID is a good predictor of purchase.  We need to keep that for predictions anyway. 
    # Second thoughts: we can predict fine without Product_ID. Is the error reduced if we use it though?
# C) Yes, men purchase items for a larger amount.
# D) Looked at in detail, the older the customers get, the more they (can) spend ... except for people above 55 
    # (guessing here, that people in retirement have less spending power)!
# E) Taking a better look at Occupation here: it matters! Code 17 has the highest purchase amount. 
# Difference between lowest and highest is a little above 1100 $, some are equal in Purchase or very close.
# F) As above we can see that the City Category is an important factor
# G) The effect of Stay_In_Current_City_Years on Purchase is noticeable, but not large in relative terms.
# H) Marital_Status makes practically no difference
# I-K) Product_Categories! They make a difference and might be very important. I) has no missing values, J) and K) have many missing values. 
    # we can impute 0 for those missing values and introduce a dummy variable 1 or / "U" as in Unknown for those that have a missing 
    # value later. The missing values might tell us that this product had no other subcategories, as we can assume 2 and 3 are.
    # Not sure if introducing the dummy variable will make a difference though. 
    # Categories range from 1 to 20 in category 1, 3 to 18 in category 2 and 3 to 18 in category 3 --> see below
    # We can use the zero for the "new" category value for the missing values.
#%%
def correlation_heatmap(df):
    hm , ax = plt.subplots(figsize=(10,12))
    colormap = sns.diverging_palette(220,10, as_cmap = True)
        
    hm = sns.heatmap(
            data=df.corr(),
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
correlation_heatmap(data1[data1_code_purchase])

# we see some weak correlations with the target, product categories 1 and 2 and product ID have a big influence
#%%
correlations = {}
features = data1_code # without purchase
for f in features:
    data_temp = data1[[f, target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0] #try non linear correlation as well
data_correlations = pd.DataFrame(correlations, index = ["Value"]).T #T transposes the column names to index and the values to the Value-ex-index
data_correlations.loc[data_correlations["Value"].abs().sort_values(ascending = False).index] #gets the original values sorted by abs()

# we see weak linear correlations. How about non linear correlations?

# TRY THIS OUT ########################### nonlinear
#%%
def vs_distribution(data,target):
    """
    Visualization code for displaying skewed distributions of features
    Put in data with all predictors AND the target column
    """
    # Create figure
    fig = plt.figure(figsize = (16,12));

    # Skewed feature plotting as histograms
    if target in data.columns:
        loop = data.drop([target],axis=1).columns
    else:
        loop = data.columns
    for i, feature in enumerate(loop):
        if np.any(pd.isnull(data[feature])):
            print("nan!") 
            data[feature].fillna(0, inplace=True)
        ax = fig.add_subplot(5, 5, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s'"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        #ax.set_ylim((0, 2000))
        #ax.set_yticks([0, 500, 1000, 1500, 2000])
        #ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])
        fig.suptitle("Distributions Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()
vs_distribution(data1[data1_code_purchase],target)
#%%
# are train and test data similar? 
# need to use chi squared test!? We have all categories...
# our expected count is the train data, our observed data is the test data and we compare them to each other
#%%
print(data_val["Product_Category_1"].value_counts().sort_index()) 
# categories 19 and 20 of prod cat 1 are missing from test
print(data1["Product_Category_1"].value_counts().sort_index())
#%%
# Chi-Squared Goodness-Of-Fit Test
l = []
coltrack = []
data = data1[data1_code]
datat = data_val[data1_code]
for col in datat.columns:
    exp = np.array( data1[col].value_counts())
    obs = np.array( data_val[col].value_counts())
    if len(exp) == len(obs):
        coltrack.append(col)
        crit = sp.stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                             df = len(exp)) # Df = number of variable categories - 1
        l.append((sp.stats.chisquare(obs, exp), crit))
pd.DataFrame(l, columns = ["chi^2 statistic and p-value","critical value"], index = coltrack).sort_values("critical value", ascending=False)

# Since our chi-squared statistic exceeds the critical value, we'd have to reject the null hypothesis that the two
# distributions are the same.
# But what does that mean? We can still do regression!?
#%%
# Chi-Squared Test of Independence as well - what is the relationship of the variables with each other? Are they independent?
# The chi-squared test of independence tests whether two categorical variables are independent. Null: they are.
# but: we would need the same number of categories in the variables themselves 
# to compare the differences in the distributions of categorical variables 
# can't use that test here

# input frequency table into contingency function
#sp.stats.chi2_contingency(exp)
#%%
data1.sample(10)
# looks good so far. 
# Define the variables we'll use for modeling
#%%
data1.info() # all the columns have the datatypes we want

#%%
# Comparison of some models
# Reduced to  3 which work fine.

## TO DO

# Might try out Ridge and Lasso later.
# what about KNN?

## TO DO

# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
       #Ensemble Methods 
       #ensemble.AdaBoostRegressor(),
       #ensemble.BaggingRegressor(),
       #ensemble.ExtraTreesRegressor(),
       #ensemble.GradientBoostingRegressor(),
       ('rfr', ensemble.RandomForestRegressor()),
       
       #Gaussian Processes
       #gaussian_process.GaussianProcessRegressor(),
       
       #GLM
       ('lr', linear_model.LinearRegression()),
       #linear_model.RidgeCV(),
       #linear_model.LassoCV(),
       #linear_model.SGDRegressor(),
       ('elnet', linear_model.ElasticNetCV()),
       ('RANSAC', linear_model.RANSACRegressor()),
       
       #Nearest Neighbor
       ('knn', neighbors.KNeighborsRegressor()),
       
       #SVM
       #svm.SVR(),
       #svm.NuSVR(),
       #svm.LinearSVR(),
       
       #Trees
       #tree.DecisionTreeRegressor(),
       #tree.ExtraTreeRegressor(),
       
       #Discriminant Analysis
       #discriminant_analysis.LinearDiscriminantAnalysis(),
       #discriminant_analysis.QuadraticDiscriminantAnalysis(),
       
       #xgboost
       ('xgb', xgboost.XGBRegressor())
       ]
#%%
#Test CV data with KFold
target = "Purchase"

cv_split = KFold(n_splits = 5, shuffle = True, random_state = 42)
   #%%
from matplotlib.ticker import FuncFormatter   
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
        print("fitting CV folds k = {}...".format(n))
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
        

predictors = data1_code
data = data1 # imputed 0's for missing values in categories, outliers are in there

#model = RandomForestRegressor(random_state = 42)
#model = RandomForestRegressor(n_estimators = 30, max_leaf_nodes = 350, random_state = 42)  # more data helps here? regulariation helps the model
model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) # good! # 3186 rmse at 0.05 tr/te
#model = XGBRegressor(n_estimators = 1000, seed = 42, random_state = 42) # good!
#model = LinearRegression() # OK
#model = ElasticNet() 
#model = RANSACRegressor() # bad? Maybe needs hyperparameter tuning
#model = ElasticNetCV() # OK

#model = neighbors.KNeighborsRegressor() # ok needs tuning
#plot_learning_curves_CV_reg(model, data, predictors, target, suptitle = "learning curve tests", title="base model "+str(model.__class__.__name__)+" CV scores", xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.08, test_size = 0.05,  scale = True, center = True, split = "Shuffle", random_state = 42)


plot_learning_curves_CV_reg(model, data, predictors, target, suptitle = "learning curve tests", title="base model "+str(model.__class__.__name__)+" CV scores", xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.05, test_size = 0.05,  scale = False, center = False, split = "Shuffle", random_state = 42)
#%%
# split in train and test and see base performance with all predictors
data = data1_dummy
#data = data1
predictors = pred_final 
#predictors = data1_code
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data1[target], test_size=0.2, shuffle = True, random_state=42)
model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) 
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("\nRMSE XGB: ", np.sqrt(mean_squared_error(y_test,preds))) 
# 3149 RMSE for all features
# 3147 RMSE for dummies with pred_final
#%%
# look for features to remove and impact on performance
# we don't have that many and can do it manually
# use pd.get_dummies later and see if we can improve RMSE by removing (maybe) some collinearity
predictors = data1_code
model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) # 1000 estimators just takes too long for testing

for feature in predictors:
    new_pred = list(predictors)
    new_pred.remove(feature)
    plot_learning_curves_CV_reg(model, data, new_pred, target, suptitle = "learning curve w/ removed predictor: "+str(feature), title="\n\n CV scores for model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.05, test_size = 0.05, split = "Shuffle", scale = False, center = False, random_state = 42 )

# perform RFE to see if we get same results


#%%
#X_train, X_test, y_train, y_test = train_test_split(data[predictors], data1[target], test_size=0.2, shuffle = True, random_state=42)
predictors = data1_code
datatr = data1[predictors]
trval = data1[target]
# use feature selection
model = XGBRegressor(n_estimators = 1000, seed = 42, random_state = 42) 

#feature selection/elimination
model_rfe = feature_selection.RFECV(model, step = 1, scoring = "neg_mean_squared_error", cv = cv_split, verbose = True)
model_rfe.fit(datatr, trval)

#transform x&y to reduced features and fit new model
#alternative: can use pipeline to reduce fit and transform steps
X_rfe = datatr.columns.values[model_rfe.get_support()] # get the reduced feature set
rfe_results = model_selection.cross_validate(model, datatr[X_rfe], trval, cv = cv_split, scoring = ("r2", "neg_mean_squared_error"), 
                                             return_train_score=True, verbose = True, n_jobs = -1)

# print results
print("After RFE Training Shape New: ", datatr[X_rfe].shape)
print("After RFE Training Columns New: ", X_rfe)

print("After RFE Training w/data score mean: {:.2f}".format(np.sqrt(-rfe_results["train_neg_mean_squared_error"].mean())))
print("After RFE Test w/data score mean: {:.2f}".format(np.sqrt(-rfe_results["test_neg_mean_squared_error"].mean())))
print("-"*20)

#tune rfe model
# param_grid = dict(n_estimators=1000,...,....)
#rfe_tune_model = model_selection.GridSearchCV(model, param_grid=param_grid, scoring = "roc_auc", cv = kf, return_train_score=True)
#rfe_tune_model.fit(train[X_rfe], target)
#%%
rfe_preds = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2',
 'Product_Category_3', 'User_ID_code', 'Product_ID_code', 'Gender_code',
 'Age_code', 'City_Category_code', 'Stay_In_Current_City_Years_code']
#After RFE Training w/data score mean: 2698.52
#After RFE Test w/data score mean: 2709.69
#%%
# plot learning curve with feature reduced model
model = XGBRegressor(n_estimators = 1000, seed = 42, random_state = 42) 
plot_learning_curves_CV_reg(model, data1, X_rfe, target, suptitle = "learning curve with RFE", title="CV scores for model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.9, test_size = 0.1, split = "Shuffle", scale = False, center = False, random_state = 42 )
# 
#%%
model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) 
model.fit(X_train[X_rfe], y_train)
preds = model.predict(X_test[X_rfe])
print("\nRMSE for RFE XGB: ", np.sqrt(mean_squared_error(y_test,preds))) 
# 3155 with whole dataset (0.1 test) no RFE 
# 3155 with RFE lol
# do the process manually 
#%%
predictors = data1_code
data = data1
target = "Purchase"
# more test data to reduce run time. Takes veeeeery long otherwise.

def manual_feature_red(data, predictors, target):
    #X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=0.2, shuffle = True, random_state=42)
    #create table to compare MLA metrics
    MLA_columns = ["Feature removed", "Train RMSE Mean", "Test RMSE Mean", "Test RMSE 3*STD"]
    MLA_compare = pd.DataFrame(columns=MLA_columns)
    row_index = 0
    for feature in predictors:
        new_pred = list(predictors)
        new_pred.remove(feature)
        
        model = XGBRegressor(n_estimators = 100, seed = 42, random_state = 42) 
    
        cv_results = model_selection.cross_validate(model, data[new_pred], data[target], cv=cv_split, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True, n_jobs = -1)
        #cv_results = model_selection.cross_validate(model, X_train[new_pred], y_train, cv=3, scoring = "neg_mean_squared_error", return_train_score=True)
        
        #fit models?
        ##model.fit(X_train, y_train)
        ##preds = model.predict(X_test)
        #model.fit(X_train[new_pred], y_train)
        #preds = model.predict(X_test[new_pred])
        #rmse = np.sqrt(mean_squared_error(y_test,preds))
        
        MLA_compare.loc[row_index, "Feature removed"] = feature
        MLA_compare.loc[row_index, "Train RMSE Mean"] = np.sqrt(cv_results["train_neg_mean_squared_error"].mean()*(-1))
        MLA_compare.loc[row_index, "Test RMSE Mean"] = np.sqrt(cv_results["test_neg_mean_squared_error"].mean()*(-1))
        #if this is a non-bias random sample, then +/-3 std from the mean, should statistically capture 99,7% of the subsets
        MLA_compare.loc[row_index, "Test RMSE 3*STD"] = np.sqrt(cv_results["test_neg_mean_squared_error"].std()*(3))
        #MLA_compare.loc[row_index, "RMSE on test set"] = rmse

        row_index += 1
        print("\nFeature {} done.".format(feature))
    MLA_compare.sort_values(by= ["Test RMSE Mean"], ascending = True, inplace=True)
    
    return MLA_compare
#%%
pd.set_option("display.max_columns",13)
MLA_compare = manual_feature_red(data, predictors, target)
MLA_compare
# dropping Occupation decreases RMSE on test set to 2925 - lower value as with the test above and full predictors. 
# We can remove what doesn't have an impact!
#%%
pred_test = ['Product_Category_1',
 'Product_Category_2',
 'Product_Category_3',
 'User_ID_code',
 'Product_ID_code',
 'Gender_code',
 'City_Category_code'] 
#%%
pred_final = ['Product_Category_1',
 'Product_Category_3',
 'Product_ID_code',
 'City_Category_code']

#%%
# split in train and test and see base performance with pred_final!
X_train, X_test, y_train, y_test = train_test_split(data[pred_final], data[target], test_size=0.2, shuffle = True, random_state=42)
model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) 
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("\nRMSE for RFE XGB: ", np.sqrt(mean_squared_error(y_test,preds))) 
# 3147 RMSE - improved!
#%%
# take a look at the learning curve
model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) 
plot_learning_curves_CV_reg(model, data, pred_final, target, suptitle = "learning curve tests", title="base model "+str(model.__class__.__name__)+" CV scores", xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.9, test_size = 0.1,  scale = False, center = False, split = "Shuffle", random_state = 42)

# great! - we decrease error with sample size. Hyperparameter tuning might help, also stacking of models might serve us well.
 # see if limiting max purchase value to 99.9 percentile decreases RMSE
#%%
# large outliers in purchase? check!
print(np.percentile(data1["Purchase"], 99.9))
#%%
# visual check
plt.figure(figsize=(16,12))
plt.plot([x for x in range(0, train.shape[0])],train[target])
plt.ylim(23500, 24000)
# seems we have a few values above 23729
#%%
biggest_amounts = data1["Purchase"][data1[target] > np.percentile(data1["Purchase"], 99.9)].count() # 150 rows with Purchase amount bigger than 23900
print("\nThe biggest amounts number {} and those are {}% of the data.".format(biggest_amounts, round(biggest_amounts*100/data1.shape[0],3)))
#%%
# split in train and test and see base performance with pred_final! and limit of Purchase
#X_train, X_test, y_train, y_test = train_test_split(data[pred_final], data[target], test_size=0.2, shuffle = True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data[data1_code], data[target], test_size=0.2, shuffle = True, random_state=42)

# RMSE purchase boundaries - we can try decreasing the upper boundaries and test it with CV
max_purchase = np.percentile(data1["Purchase"], 99.9) # 99.9 percentile is the upper boundary for purchase amount
y_train[y_train > max_purchase] = max_purchase

model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) 
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("\nRMSE for RFE XGB: ", np.sqrt(mean_squared_error(y_test,preds))) 
# 3147.85449864 RMSE with max and tested on test pred_final
# 3147.73106565 RMSE without max on test pred_final
# 3149.77435084 RMSE with max on test data1_code
# 3149 RMSE without max on test data1_code --> ergo we don't limit the max value 
#(in theory, since we can't see the test data. But we will test out the effect on submit RMSE anyway)
#... and using max_purchase increased RMSE slightly in submissions. We can trust CV here, although it was only 1 fold :D 
# Should do it with 5 and shuffle...
#%%
data1.sample(3)
#%%
def CV_target_max_purchase(estimator, data = None, predictors= None, target = None,
	                          percentile = None, k = 10, train_size= 0.9, test_size = 0.1, split = "KFold", 
                              random_state = 42, scale = False, center = False):

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
    
    # get the indices for max value percentile
    if percentile != None:
        max_purchase = np.percentile(data[target], percentile) # we set 99.9 percentile as upper boundary for purchase amount and check change in rmse
    
    # iterate through the folds and fit / predict on train/val folds
    for n, (train_i, test_i) in enumerate(cv.split(data[predictors], data[target])): # use kfold to "average" over the whole dataset and compute a smoothed out learning curve
        print("fitting CV folds k = {}...".format(n))
        X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
        y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]
        
        if percentile != None:
        # apply the max value percentile to target in train only
            y_train.loc[y_train["Purchase"] > max_purchase] = max_purchase
        
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
        ravel_y_train = np.array(y_train).ravel()
        estimator.fit(X_train.iloc
                ,ravel_y_train) # ravel the vector
        
        #calculate the training accuracy only using those training examples
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
data = data1
predictors = data1_code
target = "Purchase"

model = XGBRegressor(
 learning_rate = 0.1,
 n_estimators = 6628,
 max_depth=5,
 min_child_weight=15,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 n_jobs= -1,
 scale_pos_weight=1,
 seed=42,
 random_state = 42)

train_rmse, test_rmse, train_r2, test_r2 = CV_target_max_purchase(model, data, predictors, target, percentile = None, k = 5)
values = (train_rmse, test_rmse, train_r2, test_r2)
prints = ["Train RMSE", "Test RMSE", "Train r2", "Test r2"]

for result, value in zip(prints, values):
    print("\nResult for ", result, " is: ", value)
#%%
train_rmse, test_rmse, train_r2, test_r2 = CV_target_max_purchase(data, predictors, target, percentile = 99.9, k = 5)
values = (train_rmse, test_rmse, train_r2, test_r2)
prints = ["Train RMSE", "Test RMSE", "Train r2", "Test r2"]

for result, value in zip(prints, values):
    print("\nResult for ", result, " is: ", value)

#%%
# take a look at the learning curve
model = XGBRegressor(n_estimators = 30, seed = 42, random_state = 42) 
plot_learning_curves_CV_reg(model, data, pred_final, target, suptitle = "learning curve tests", title="base model "+str(model.__class__.__name__)+" CV scores", xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.9, test_size = 0.1,  scale = False, center = False, split = "Shuffle", random_state = 42)
#%%
# Purchase boundaries - set them to try it out - no decrease in rmse
#max_purchase = np.percentile(data1["Purchase"], 99.9) # 99.9 percentile is the upper boundary for purchase amount
#data1.loc[train["Purchase"] > max_purchase, "Purchase"] = max_purchase

#%%
# compare learning curves for some regressors:
for name,model in MLA:
    plot_learning_curves_CV_reg(model, data, predictors, target, suptitle = "learning curve tests", title="Base model CV scores for model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="RMSE", ylabel2="r2 score", k = 5, granularity = 5, train_size = 0.1, test_size = 0.1, split = "Shuffle", scale = False, center = False, random_state = 42 )
#xgb seems to be a very sensible choice already
# lr has high rmse, but consistent
# rfr might be a good choice tuned
# ElasticNetCV or ElasticNet might also work
#%%
def sklearn_learning_curves_reg(model, data, predictors, target, suptitle = "", title="", xlabel="", ylabel="", k = 10, granularity = 10, train_size = 0.8, test_size = 0.2, random_state = 42):
    
    #initiate cv method
    cv = ShuffleSplit(n_splits= k , train_size = train_size, test_size= test_size, random_state=random_state)
    
    # set the max train samples according to test size
    max_train_samples = len(data)*train_size
    
    # use sklearn learning_curve function to compute learning curves with cv method and data
    training_set_sizes = np.linspace(10, max_train_samples, granularity, dtype='int')
    train_sizes, train_scores, valid_scores = learning_curve(model, 
                   data[predictors], np.array(data[target]).ravel(), train_sizes = training_set_sizes, cv = cv, 
                   shuffle = True, scoring = "neg_mean_squared_error", random_state = random_state)
  
    train_scores = np.sqrt(-train_scores) #make positive RMSE errors
    valid_scores = np.sqrt(-valid_scores)
    
    #stds
    train_std = np.std(train_scores, axis = 1)
    valid_std = np.std(valid_scores, axis = 1)

    # compute means of the k splits
    train_scores = np.mean(train_scores, axis = 1)
    valid_scores = np.mean(valid_scores, axis = 1)
    
    # plot learning curves on different charts 
    fig, ax = plt.subplots(figsize=(12, 6))
    # plot the train / test score std ranges
    ax.fill_between(train_sizes, train_scores - train_std, train_scores + train_std, alpha = 0.15, color='gold')
    ax.fill_between(train_sizes, valid_scores - valid_std, valid_scores + valid_std, alpha = 0.15, color='steelblue')
    # plot the scores
    ax.plot(train_sizes, train_scores, c='gold', marker = "o")
    ax.plot(train_sizes, valid_scores, c='steelblue', marker = "o")
    
    # format the charts to make them look nice
    fig.suptitle(suptitle, fontweight='bold', fontsize='20')
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel, size=16)
    ax.set_ylabel(ylabel, size=16)
    ax.legend(['training set', 'validation set'], fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(min(valid_scores[2:])-min(valid_scores[2:])/10, (max(valid_scores[2:])+max(valid_scores[2:])/10))
    ax.grid(b=True)
    for x,y in zip(train_sizes,valid_scores):
        ax.annotate(str(round(y,3)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')

predictors = data1_code
data = data1 # imputed 0's for missing values in categories, outliers are in there

#model = RandomForestRegressor(random_state = 42)
#model = RandomForestRegressor(max_leaf_nodes = 350, random_state = 42)
#model = XGBRegressor(seed = 42, random_state = 42)
#model = LinearRegression()
#model = ElasticNet()
#model = RANSACRegressor()
#model = ElasticNetCV()

#sklearn_learning_curves_reg(model, data, predictors, target, suptitle = "learning curve tests", title="all features, model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="RMSE", train_size = 0.05, test_size = 0.05, k = 5, granularity = 5)
#sklearn_learning_curves_reg(model, data, predictors, target, suptitle = "learning curve tests", title="all features, model: "+str(name), xlabel="n samples", ylabel="RMSE", train_size = 0.008, test_size = 0.005, k = 10, granularity = 10)
#%%
data_val.info()
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
    cv_results = model_selection.cross_validate(alg, data1[predictors], data[target], cv=cv_split, scoring = "neg_mean_squared_error", return_train_score=True, n_jobs = -1)

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
sns.barplot(x="MLA Test  RMSE Mean", y="MLA Name", data=MLA_compare, color="m")
plt.title("MLA RMSE Score\n")
plt.xlabel("RMSE Score")
plt.ylabel("Algorithm")
#%%
# First: get dummies from our LabelEncoded columns.
data1_dummy = pd.get_dummies(data = data1, columns=pred_final, drop_first=True)
data1_dummy = pd.concat([data1_dummy, data1[target]], axis=1)

#%%
data1_dummies1 = list(data1.drop(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category',
       'Stay_In_Current_City_Years', 'Purchase'],axis=1).columns.values)
# use that for comparison with SKB RMSE of 2915
    
data1_dummies2 = list(data1.drop(['Gender', 'Age', 'City_Category',
   'Stay_In_Current_City_Years', 'Purchase'],axis=1).columns.values)

data1_dummies3 = list(data1.drop(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category',
       'Stay_In_Current_City_Years', 'Purchase'],axis=1).columns.values)

data1_dummies_ID = list(data1.drop(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category',
   'Stay_In_Current_City_Years', 'Purchase'],axis=1).columns.values)
#%%
data1[data1_dummies3].info()
# 45 MB size here
# looks fine
#%%
# Tests with this:
# Reduce the data set size and see what difference it makes in computing and rmse
ssplit = ShuffleSplit(n_splits=5, test_size= .05, train_size = .1, random_state=0)

#%%
reg = XGBRegressor()
reg.fit(data1[predictors], data1[target])
predictions = reg.predict(data1[predictors])
print("\nRMSE for full train set: ",np.sqrt(mean_squared_error(data1[target], predictions)))
#2917
#%%

### Cleaning of outliers based on Squared Errors ###

# compare model performance before and after that step and see if test score improved #

def outlierCleaner(predictions, data, predictors, target, deletefraction = 0.01):
     """
         Clean away the fraction% of points that have the largest
         residual errors (difference between the prediction
         and the actual net worth).

         Return a DataFrame named cleaned_data where
         a column error was inserted with the computed squared errors 
         between predictions and target values.
     """
     cleaned_data = data.copy(deep = True)
     
     for i in range(len(predictions)):
         cleaned_data["errors"] = (cleaned_data[target].iloc[i]-predictions[i])**2
     print("sorting now")
     cleaned_data.sort_values(by = "errors", kind = "mergesort", ascending=False, inplace=True)
     return cleaned_data.iloc[int((cleaned_data).shape[0]*deletefraction):]
#%%
#X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], train_size = 0.8, test_size=0.2, shuffle=True, 
#                                                 random_state= 42) 
cleaned_data = outlierCleaner(predictions, data1, predictors, target, deletefraction = 0.01)
#%%
reg2 = XGBRegressor()
reg2.fit(cleaned_data[predictors], cleaned_data[target])
predictions2 = reg.predict(cleaned_data[predictors])
print("\nRMSE for train set w/o outliers: ",np.sqrt(mean_squared_error(cleaned_data[target], predictions2)))
# RMSE 2930 - removal does not help 
#%%
### Gridsearch for our Meta Stacking Ensemble ###
#Hyperparameter tune with GridSeachCV
data = data1
target = "Purchase"
random_state = 42

grid_n_estimator = [5,10,50,100,150,300,2000]
grid_ratio = [0.03, .1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_coeffs = [0, 0.1, 1, 10]
grid_gamma = [0.001, 0.01, 0.1, 1, "auto"]
grid_max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]
#grid_min_leaf_samples = [1, 3, 5]
grid_min_samples = [5, 10, .03, .05, .10]
grid_degree = [2, 3, 4]
grid_criterion = ["gini", "entropy"]
grid_bool = [True, False]
grid_seed = [42]
grid_regular = [0.00001, 0.0001, 0.001, 0.01, 0.1]
grid_features = [2,4,6,8, np.sqrt(len(predictors)), len(predictors)]

grid_param_level2 = [
      [{
      #SVR
      "kernel": ["linear", "poly", "rbf", "sigmoid"],
      "C": [0.01, 0.1, 1, 10, 100, 1000], 
      "gamma": ["auto"],
      "degree": grid_degree,
      #"tol" : grid_regular,
      #"coef0": grid_coeffs,
      "epsilon": [0.01, 0.1, 0.3],
      "random_state": grid_seed,
      "verbose" :  [True]
      }],  
        
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
        ]


grid_param_level1 = [
        
      [{
      #LinearRegression
      "fit_intercept": grid_bool, 
      }],
        
          [{ 
      #TheilSenRegressor
      "max_subpopulation": [10000], 
      "n_subsamples": [None],
      "max_iter": [2000],
      "tol": [0.0001, 0.001, 0.01],
      "random_state": grid_seed,
      "verbose": [True],
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

     #   [{
      #ExtraTreesRegressor
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
      #RandomForestRegressor
      "max_features" : grid_features,
      "n_estimators": grid_n_estimator, #def = 10
      "criterion": grid_criterion, #def = "gini"
      "max_depth": grid_max_depth, #def = None
      "oob_score": [True], #def = False
      "min_samples_split" : grid_min_samples,
      #"min_samples_leaf" : grid_min_leaf_samples,
      "random_state": grid_seed
      }],
  
    [{
      #KNeighborsRegressor
      "n_neighbors": [x for x in range(1,51)], 
      "weights": ["uniform", "distance"], 
      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
      }],   
   ]
#%%
MLA1 = [
        #("etc", ensemble.ExtraTreesRegressor()),
        ("rfc", ensemble.RandomForestRegressor()),
        ("knn", neighbors.KNeighborsRegressor()),
    ]  

MLA2 = [
        #GLM
       ("lr", linear_model.LinearRegression()),
       ("ridge", linear_model.TheilSenRegressor()),
       ("elnet", linear_model.ElasticNet()),
       ("RANSAC", linear_model.RANSACRegressor()),
       ]    

MLA3 = [
       #SVR
       ('svr', SVR())
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
        best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv, scoring = scoring)
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
predictors_scale = data1_code # all feature names
target = "Purchase" # response name

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaled = scaler.fit(dataset[predictors_scale])
scaled_data = scaled.transform(dataset[predictors_scale])
scaled_data = pd.DataFrame(scaled_data, columns = predictors_scale )
scaled_data = pd.concat([scaled_data, dataset[target]], axis=1)
#%%
gridsearch_params_mla(MLA1, grid_param_level0, scaled_data, predictors, target, cv = cv_split, scoring = "neg_mean_squared_error")
#%%
gridsearch_params_mla(MLA2, grid_param_level1, scaled_data, predictors, target, cv = cv_split, scoring = "neg_mean_squared_error")
#%%
gridsearch_params_mla(MLA3, grid_param_level2, scaled_data, predictors, target, cv = cv_split, scoring = "neg_mean_squared_error")
#%%
data = data1
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], shuffle = True, test_size=0.3, random_state=42)

model = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 n_jobs= -1,
 scale_pos_weight=1,
 seed=27)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["rmse"], eval_set=eval_set, verbose=True, early_stopping_rounds=100)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
print("R^2 score: %.3g" % r2_score(y_test, predictions))
print("explained variance score: %f" % metrics.explained_variance_score(y_test, predictions))
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# plot regression error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('rmse')
plt.title('XGBoost rmse')
plt.show()

# best number of estimators found: 1000. lol. rmse of 2578
#%%
cv_split.get_n_splits()
#%%
data = data1
#predictors = pred_final
#predictors = pred_test
predictors = data1_code
# Function for tuning XGB hyperparameters. 
# y defaults to our defined target variable.

def modelfit_reg(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, folds = None, early_stopping_rounds=100, y=target):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgboost.DMatrix(dtrain[predictors].values, label=y.values)
        cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds, stratified = False, verbose_eval= True, seed = 42, folds = folds)
        alg.set_params(n_estimators= int(cvresult.shape[0]/(1-1/folds.get_n_splits())))
        print("\nBest iteration: ",cvresult.shape[0])
        print("optimal number of estimators is: ", int(cvresult.shape[0]/(1-1/folds.get_n_splits())))
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y ,eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
        
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, dtrain_predictions))) 
    print("explained variance score: %f" % metrics.explained_variance_score(y, dtrain_predictions))
    print(pd.Series(alg.get_booster().get_score(importance_type='weight')).sort_values(ascending=False))
    feat_imp = pd.Series(alg.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    pyl.ylabel('Feature Importance Score')

xgb_reg1 = XGBRegressor(
 learning_rate = 0.1,
 n_estimators = 30,
 max_depth=5,
 min_child_weight=15,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 n_jobs= -1,
 scale_pos_weight=1,
 seed=42,
 random_state = 42)
modelfit_reg(xgb_reg1, data, predictors, y=data[target], folds = cv_split)

#RMSE : 2522
#[999]   train-rmse:2512.38+3.98402      test-rmse:2569.18+6.77892 with full features - 2820 submit RMSE

# rmse for reduced features is higher
#[999]   train-rmse:2651.66+2.66277      test-rmse:2672.1+9.87238 
#RMSE : 2654

# rmse for pred_test
#[999]   train-rmse:2540.39+2.56445      test-rmse:2587.7+9.95952
#RMSE : 2545

#RMSE : 2276
#[6628]  train-rmse:2247.02+8.28422      test-rmse:2477.3+34.8273 ---> 8285 estimators
#explained variance score: 0.794660
#%%
# use random grid search to search for best hyperparameters
# create hyperparameter grid to search through
# number of trees in xgb
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)]
# number of features to consider at every split
max_features = ["auto", "sqrt"]
# maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10,110, num = 11)]
max_depth.append(None)
#minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(2,50,10)]
# minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(1, 15, 5)]
# method of selecting samples for training each tree
#bootstrap = [True, False] # only for rf 
#weight of child node in xgb
min_child_weight = [int(x) for x in np.linspace(1,10,10)]
#ratio of total samples to choose from for tree building step
subsample = [x/100 for x in np.linspace(30,100,10)]
#ratio of features to choose from for tree building
colsample_bytree = [x/100 for x in np.linspace(30,100,10)]
# reg param 1
gamma = [x/100.0 for x in np.linspace(0,30,10)]
# reg param 2 
reg_alpha = [x/1000000 for x in np.logspace(1,8,5)]
# reg param 3
reg_lambda = [x/1000000 for x in np.logspace(1,8,5)]
# learning rate for the trees and preds 
learning_rate = [x/100000 for x in np.linspace(1,10000,5)]

#'gamma':[i/100.0 for i in range(0,20)]
#'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
#'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
#"learning_rate": [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],

#create the random_grid
random_grid = dict(n_estimators = n_estimators,
                   max_depth=max_depth,
                   min_child_weight = min_child_weight,
                   subsample = subsample,
                   colsample_bytree = colsample_bytree,
                   gamma = gamma,
                   reg_alpha = reg_alpha,
                   reg_lambda = reg_lambda,
                   learning_rate = learning_rate                   )
print(random_grid)

#%%
data = data1
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], shuffle = True, test_size=0.1, random_state=42)

# create base model to tune
xgb_reg1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 n_jobs= -1,
 scale_pos_weight=1,
 seed=42,
 random_state = 42)
# random search for parameters, using 3 fold cv
# search across 100 different combinations, and use all available cores
xgb_random = RandomizedSearchCV(estimator = xgb_reg1, param_distributions = random_grid, n_iter = 2, cv = 3, verbose = 2, random_state = 42#, n_jobs = 2
                                )
# fit the random search # covering n_iter combinations instead of every combination
#xgb_random.fit(data[predictors], data[target]) # fit on full set for real param search without predictions
xgb_random.fit(X_train, y_train) # we do this here so we can compare fitted estimators
#results:
# phew, 78 minutes for cv = 3 and n_iter = 2 with 70% of the dataset. That takes a seriously long time. 
# Tuning by hand with GridSearches for certain parameters might be faster
print(xgb_random.best_params_)
# param_grid_random = {'subsample': 0.76666666666666661, 'reg_lambda': 1.778279410038923, 'reg_alpha': 1.0000000000000001e-05, 'n_estimators': 733, 'min_child_weight': 9, 'max_depth': 80, 'learning_rate': 0.050005000000000001, 'gamma': 0.033333333333333333, 'colsample_bytree': 0.53333333333333333}
#narrow down the ranges from that with "manual" gridsearch
#evaluate result first and compare with base model
#%%
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    print("\nModel Performance")
    print("\Average Error: {:0.4f}$.".format(np.mean(errors)))
    print("Accuracy = {:0.2f}%.".format(accuracy))
    print("\nRMSE = {:0.2f}.".format(rmse))
    return accuracy, rmse

base_model = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 n_jobs= -1,
 scale_pos_weight=1,
 seed=42,
 random_state = 42)

base_model.fit(X_train, y_train)
base_accuracy, base_rmse = evaluate(base_model, X_test, y_test)

# param_grid_random = {'subsample': 0.76666666666666661, 'reg_lambda': 1.778279410038923, 'reg_alpha': 1.0000000000000001e-05, 'n_estimators': 733, 'min_child_weight': 9, 'max_depth': 80, 'learning_rate': 0.050005000000000001, 'gamma': 0.033333333333333333, 'colsample_bytree': 0.53333333333333333}
best_random = xgb_random.best_estimator_ # assign the best hyperparameters of randomized search 
random_accuracy, random_rmse = evaluate(best_random, X_test, y_test)

print("Improvement of {:0.2f}% and {:0.2f}$ in RMSE.".format(100* (random_accuracy-base_accuracy)/base_accuracy, (base_rmse-random_rmse)))
# we didnt improve the rmse. Parameters we randomly used weren't better, which is not to be expected for just 2 runs with such a huge range
# In[131]:

#Step 2 of modelfit_reg for xgb: Tune max_depth and min_child_weight first

n_estimators=2000    

max_depth = [int(x) for x in np.linspace(3,110, num = 4)]
max_depth.append(None)
min_child_weight = [int(x) for x in np.linspace(1,30,4)]

# best results so far: max_depth 3, min_child_weight 20

param_test1 = {
 'max_depth': max_depth,
 'min_child_weight': min_child_weight
 }
gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators=n_estimators, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear',  scale_pos_weight=1, seed=42, random_state = 42), 
 param_grid = param_test1, scoring='neg_mean_squared_error', n_jobs= 1,iid=False, cv=cv_split, verbose = 10)
gsearch1.fit(data[predictors],data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#%%
n_estimators = 2000

max_depth = 3
#  {'min_child_weight': 15} best score
min_child_weight = [int(x) for x in np.linspace(15,25,4)]

param_test1a = {
 'min_child_weight': min_child_weight
}
gsearch1a = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators=n_estimators, max_depth=max_depth,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear',  scale_pos_weight=1, seed=42, random_state = 42), 
 param_grid = param_test1a, scoring='neg_mean_squared_error',n_jobs= 1,iid=False, cv=cv_split, verbose = 10)
gsearch1a.fit(data[predictors],data[target])
gsearch1a.cv_results_, gsearch1a.best_params_, gsearch1a.best_score_
#%%
max_depth = 3

min_child_weight = [x for x in range(13,18,2)]

param_test1b = {
 'min_child_weight': min_child_weight
}
gsearch1b = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators=n_estimators, max_depth=max_depth,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear',  scale_pos_weight=1, seed=42, random_state = 42), 
 param_grid = param_test1b, scoring='neg_mean_squared_error',n_jobs= 1,iid=False, cv=cv_split, verbose = 10)
gsearch1b.fit(data[predictors],data[target])
gsearch1b.cv_results_, gsearch1b.best_params_, gsearch1b.best_score_
# best: 15
#%%
min_child_weight = 15

n_estimators = 6628

max_depth = [int(x) for x in np.linspace(5,16, num = 3)]
max_depth.append(None)

param_test1c = {
 'max_depth': max_depth,
}
gsearch1c = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators=n_estimators, max_depth=5,
 min_child_weight=min_child_weight, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear',  scale_pos_weight=1, seed=42, random_state = 42), 
 param_grid = param_test1c, scoring='neg_mean_squared_error',n_jobs= 1,iid=False, cv=cv_split, verbose = 10)
gsearch1c.fit(data[predictors],data[target])
gsearch1c.cv_results_, gsearch1c.best_params_, gsearch1c.best_score_
# In[92]:


#{'max_depth': 5, 'min_child_weight': 5},
# We’ll search for values above and below the optimum values
# because we took an interval.
param_test2 = {
 'max_depth':[3,4,5,6,7],
 'min_child_weight':[3,4,5,6,7,8,9]
}
gsearch2 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=n_estimators, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', scale_pos_weight=1,seed=42, random_state = 42), 
 param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs= 1,iid=False, cv=cv_split)
gsearch2.fit(data[predictors],data[target])
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


# In[132]:

# {'max_depth': 3, 'min_child_weight': 8},
max_depth = 3
min_child_weight = 8

# In[133]:

#Step 3: Tune gamma
#Now lets tune gamma value using the parameters already tuned above. 

gamma = [x/100.0 for x in np.linspace(0,50,10)]

param_test3 = {
 'gamma': gamma
}
gsearch3 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators, max_depth = max_depth,
 min_child_weight = min_child_weight, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', scale_pos_weight=1,seed=42, random_state = 42), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=cv_split)
gsearch3.fit(data[predictors],data[target])
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


# In[134]:

# {'gamma': 0.17},
gamma = 0.17 

# Before proceeding, a good idea would be to re-calibrate the 
# number of boosting rounds for the updated parameters.

xgb_reg2 = XGBRegressor( #adjust values according to gridsearches
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=gamma,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 n_jobs = -1,
 scale_pos_weight=1,
seed=42, random_state = 42)
modelfit_reg(xgb_reg2, data, predictors, y=data[target])


# In[136]:

#Step 4: Tune subsample and colsample_bytree
#The next step would be try different subsample and colsample_bytree values. 

#Lets do this in 2 stages as well and take values 0.6,0.7,0.8,0.9 
# for both to start with.

#ratio of total samples to choose from for tree building step
subsample = [x/100 for x in np.linspace(30,100,10)]
#ratio of features to choose from for tree building
colsample_bytree = [x/100 for x in np.linspace(30,100,10)]

n_estimators2 = 24

param_test4 = {
 'subsample': subsample,
 'colsample_bytree': colsample_bytree
}
gsearch4 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear',  scale_pos_weight=1,seed=42, random_state = 42), 
 param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch4.fit(data[predictors],data[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[137]:


#{'colsample_bytree': 0.7, 'subsample': 0.7}
# Now we should try values in 0.05 interval around these.

param_test5 = {
 'subsample':[i/100.0 for i in range(45,80,5)],
 'colsample_bytree':[i/100.0 for i in range(45,80,5)]
}
gsearch5 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear',  scale_pos_weight=1,seed=42, random_state = 42), 
 param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch5.fit(data[predictors],data[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[138]:

#{'colsample_bytree': 0.7, 'subsample': 0.7},
colsample_bytree = 0.5
subsample = 0.6

# Step 5: Tuning Regularization Parameters

# Next step is to apply regularization to reduce overfitting. 

reg_alpha = [x/1000000 for x in np.logspace(1,8,5)]

param_test6 = {
 'reg_alpha': reg_alpha
}
gsearch6 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
 objective= 'reg:linear', scale_pos_weight=1,seed=42, random_state = 42, reg_alpha = 1),  
 param_grid = param_test6, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch6.fit(data[predictors],data[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[139]:


#{'reg_alpha': 1},

param_test7 = {
# 'reg_alpha':[0, 0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
# 'reg_alpha':[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
 'reg_alpha':[0.1, 0.3, 1, 3, 10, 30, 90]
}
gsearch7 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
 objective= 'reg:linear', scale_pos_weight=1,seed=42, random_state = 42, reg_alpha = 1), 
 param_grid = param_test7, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch7.fit(data[predictors],data[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


# In[140]:


#{'reg_alpha': 0.0},
reg_alpha =  0.3


xgb_reg3 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=gamma,
 subsample=subsample, 
 colsample_bytree=colsample_bytree,
 reg_alpha=reg_alpha,
 objective= 'reg:linear',
 n_jobs=-1,
 scale_pos_weight=1,
seed=42, random_state = 42) 
modelfit_reg(xgb_reg3, data, predictors, y=data[target])


# In[141]:

n_estimators2 = 22

reg_lambda = [x/1000000 for x in np.logspace(1,8,5)]

param_test6 = {
 'reg_lambda':reg_lambda
}
gsearch6 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, reg_alpha = reg_alpha, subsample=subsample, colsample_bytree=colsample_bytree,
 objective= 'reg:linear',  scale_pos_weight=1 , seed=42, random_state = 42, reg_lambda = 1), 
 param_grid = param_test6, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch6.fit(data[predictors],data[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[142]:

# narrow lambda 

#{'reg_lambda': 1},
param_test6l = {
 #'reg_lambda':[0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
 #'reg_lambda':[0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
 'reg_lambda':[0.03, 0.1, 0.3, 1, 3, 10, 20, 30, 50, 90, 100, 300]
}
gsearch6l = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, reg_alpha = reg_alpha, subsample=subsample, colsample_bytree=colsample_bytree,
 objective= 'reg:linear',  scale_pos_weight=1,seed=42, random_state = 42, reg_lambda = 1), 
 param_grid = param_test6l, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch6l.fit(data[predictors],data[target])
gsearch6l.grid_scores_, gsearch6l.best_params_, gsearch6l.best_score_


# In[143]:


#{'reg_lambda': 3},
reg_lambda = 1


xgb_reg4 = XGBRegressor(
 learning_rate = 0.1,
 n_estimators = 1000,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=gamma,
 subsample=subsample, 
 colsample_bytree=colsample_bytree,
 reg_alpha = reg_alpha,
 reg_lambda= reg_lambda,
 objective= 'reg:linear',
 n_jobs = -1,
 scale_pos_weight=1,
 seed=42, random_state = 42)
modelfit_reg(xgb_reg4, data, predictors, y=data[target])


# In[151]:
n_estimators2 = 22

learning_rate = [x/100000 for x in np.linspace(1,10000,5)]

param_test6a = {
        "learning_rate": learning_rate,  #tune learning rate with max estimators first, then adjust trees to avoid overfitting
        #"n_estimators": [int(i/100) for i in range(100,4000,100)]
        }
gsearch6a = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, reg_alpha = reg_alpha, subsample=subsample, colsample_bytree=colsample_bytree,
 objective= 'reg:linear', scale_pos_weight=1,seed=42, random_state = 42, reg_lambda = reg_lambda), 
 param_grid = param_test6a, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch6a.fit(data[predictors],data[target])
gsearch6a.grid_scores_, gsearch6a.best_params_, gsearch6a.best_score_
#%%
# narrow learning_rate window
learning_rate = [x/100000 for x in np.linspace(1,10000,5)]

param_test6b = {
        "learning_rate": learning_rate,  
        }
gsearch6b = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=n_estimators2, max_depth=max_depth,
 min_child_weight=min_child_weight, gamma=gamma, reg_alpha = reg_alpha, subsample=subsample, colsample_bytree=colsample_bytree,
 objective= 'reg:linear', scale_pos_weight=1,seed=42, random_state = 42, reg_lambda = reg_lambda), 
 param_grid = param_test6b, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=3)
gsearch6b.fit(data[predictors],data[target])
gsearch6b.grid_scores_, gsearch6b.best_params_, gsearch6b.best_score_

# In[152]:

#{'learning_rate': 0.03, 'n_estimators': 31},

# Step 6b: fitting the estimator a last time
learning_rate = 0.1
n_estimators = 1000 # do the fit with max cv estimators out of the modelfit function


xgb_reg5 = XGBRegressor(
 learning_rate = learning_rate,
 n_estimators=n_estimators,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=gamma,
 subsample=subsample, 
 colsample_bytree=colsample_bytree,
 reg_alpha = reg_alpha,
 reg_lambda= reg_lambda,
 objective= 'reg:linear',
 n_jobs = -1,
 scale_pos_weight=1,
 seed=42, random_state = 42)
modelfit_reg(xgb_reg5, data, predictors, y=data[target])


# In[153]:


xgb_pred = xgb5.predict(X_test)
xgb_pred_proba = xgb5.predict_proba(X_test)[:,1]
#Print model report:
print("\nModel Report")
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, xgb_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, xgb_pred_proba))
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))
# In[154]:
# assign full data again to train the model for test predictions
data = data1

xgb_reg6 = XGBRegressor(
 learning_rate = learning_rate,
 n_estimators=n_estimators,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=gamma,
 subsample=subsample, 
 colsample_bytree=colsample_bytree,
 reg_alpha = reg_alpha,
 reg_lambda= reg_lambda,
 objective= 'reg:linear',
 n_jobs = -1,
 scale_pos_weight=1,
 seed=42,
 random_state=42)
modelfit_reg(xgb_reg6, data, predictors, y=data[target])
#%%

# After model tuning we can ensemble the models

y = target
# assume we defined params above as param
# param = {}

num_round = 700

dtrain = data1
dtest = data_val

seeds = [1101, 2343, 3543, 4556, 5696]
test_preds = np.zeros((data_val.shape[0], len(seeds)))

for run in range(len(seeds)):
    print("\rXGB run {} / {} ".format(run+1, len(seeds)))
    param["seed"] = seeds[run]
    
    # get the data matrix for train
    xgtrain = xgboost.DMatrix(dtrain[predictors].values, label=y.values)
    #... and test
    xgtest = xgboost.DMatrix(dtest[predictors].values, missing = np.nan)
    # train xgboost
    reg = xgboost.train(param, dtrain, num_round)
    test_preds[:,run] = reg.predict(xgtest)
    
test_preds = np.mean(test_preds, axis=1)
#%%
### STACKED Meta Ensembling  ###

# need to use gridsearch on most models to tune parameters #

random_state = 42
data = data1

predictors = data1_code
target = "Purchase"
# Hold back part of train set and split into train and val set
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=0.1, shuffle = True, random_state=random_state)

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
    
    folds=StratifiedKFold(n_splits=n_fold, random_state=42)

    train_pred = np.zeros((ntrain,))    
    test_pred = np.zeros((ntest,)) 
    test_pred_skf = np.empty((n_fold, ntest))
    
    # run the models on k folds of the training set and predict on train and validation set
    # for train_indices,val_indices in folds.split(X_train,y_train.values):
    for i, (train_indices, val_indices) in enumerate(folds.split(X_train,y_train)):
        x_tr,x_val = X_train.iloc[train_indices], X_train.iloc[val_indices]
        y_tr,y_val = y_train.iloc[train_indices], y_train.iloc[val_indices]
        
        # scale the data for the algorithms that need it
        _, X_test_scaled = scaling(x_tr, X_test, scale, center)
        x_tr, x_val = scaling(x_tr, x_val, scale, center)
        
        #fit the models to every combined train subset and predict on x_val 
        model.fit(X=x_tr, y=y_tr)
        train_pred[val_indices] = model.predict(x_val)
        test_pred_skf[i, :] = model.predict(X_test_scaled)
        
    #test_pred[:] = sp.stats.mode(test_pred_skf)[0] # should we rather use max voting and return the mode?
    test_pred[:] = test_pred_skf.mean(axis=0) # mode seems to work equally well in this case, for reg we use mean anyway
    return test_pred, train_pred

def stacking_modelfit(model, X_train, X_test, y_train, folds, scale = False, center = False):
    test_pred, train_pred = Stacking(model, X_train, X_test, y_train, folds, scale, center)
    train_pred = pd.DataFrame([train_pred]).T
    test_pred = pd.DataFrame([test_pred]).T
    
    return train_pred, test_pred

#scale data now! - fit on train, transform on both
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaled = scaler.fit(X_train)
X_train = scaled.transform(X_train)
X_train = pd.DataFrame(X_train, columns = predictors)
# apply scaling to test set
X_test = scaled.transform(X_test)
X_test = pd.DataFrame(X_test, columns = predictors)
       
# level 0 
model1 =  ensemble.RandomForestRegressor(random_state = 42) # criterion = "entropy", max_depth=5, n_estimators=10, oob_score= True, 
train_pred1, test_pred1 = stacking_modelfit(model1, X_train, X_test, y_train, 10)

model2 = neighbors.KNeighborsRegressor() # algorithm= 'auto', n_neighbors= 12, weights= 'uniform' 
train_pred2, test_pred2 = stacking_modelfit(model2, X_train, X_test, y_train, 10, True, True)

model3 =  linear_model.LinearRegression()
train_pred3, test_pred3 = stacking_modelfit(model3, X_train, X_test, y_train, 10)

model4 = linear_model.ElasticNet()
train_pred4, test_pred4 = stacking_modelfit(model4, X_train, X_test, y_train, 10)

df1 = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4 ],  axis=1)
df_test1 = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4],  axis=1)

df1.columns = ["RFC", "KNN", "ETC", "GBC"]
df_test1.columns = ["RFC", "KNN", "ETC", "GBC"]

df_train = X_train.copy().reset_index(drop=True)
df_train = pd.concat([df_train, df1], axis = 1)

df_test = X_test.copy().reset_index(drop=True)
df_test = pd.concat([df_test, df_test1], axis = 1)


# level 1
# add 3 more predictions as features and then predict on those as well in the final model
model5 = linear_model.RANSACRegressor()
train_pred5, test_pred5 = stacking_modelfit(model5, df_train, df_test, y_train, 10)

#model6 = SVC() #C=1000, decision_function_shape = "ovo", degree = 2, gamma = 0.01, kernel = "rbf", probability = True, random_state= 42
model6 = SVR()
train_pred6, test_pred6 = stacking_modelfit(model6, df_train, df_test, y_train, 10)

model7 = linear_model.TheilSenRegressor()
train_pred7, test_pred7 = stacking_modelfit(model7, df_train, df_test, y_train, 10)

df2 = pd.concat([train_pred5, train_pred6, train_pred7],  axis=1)
df_test2 = pd.concat([test_pred5, test_pred6, test_pred7],  axis=1)

df2.columns = ["LR", "NuSVC", "XGB"]
df_test2.columns = ["LR", "NuSVC", "XGB"]

df_train = pd.concat([df_train, df2], axis = 1)
df_test = pd.concat([df_test, df_test2], axis = 1)

# correlations of models
correlation_heatmap(df_train)

# final level
#model = SVC(C=1000, decision_function_shape = "ovo", degree = 2, gamma = 0.01, kernel = "rbf", probability = True, random_state= 42)
#model = ensemble.RandomForestClassifier(criterion = "entropy", max_depth=5, n_estimators=10, oob_score= True, random_state = 42)
#model = linear_model.LogisticRegressionCV(fit_intercept=True, random_state= 42, solver= 'newton-cg')
model = XGBRegressor(
 learning_rate = learning_rate,
 n_estimators=n_estimators,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=gamma,
 subsample=subsample, 
 colsample_bytree=colsample_bytree,
 reg_alpha = reg_alpha,
 reg_lampbda= reg_lambda,
 objective= 'binary:logistic',
 n_jobs = -1,
 scale_pos_weight=1,
 seed=42, random_state = 42)

model.fit(df_train,y_train)
blend_pred = model.predict(df_test)
print("\nModel acc score:  %.3g" % np.sqrt(metrics.mean_squared_error(y_test,blend_pred)))
# winner is NuSVC with 88% acc!

#%%
xgb_reg1 = XGBRegressor(
 learning_rate=0.1,
 n_estimators=6628,
 max_depth=5,
 min_child_weight=15,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 n_jobs= -1,
 scale_pos_weight=1,
 seed=42,
 random_state = 42)

# we haven't limited the upper amount to the 99.9 percentile in training data. 
# Test impact of that as well next time
predictors = data1_code
#predictors = pred_final
#predictors = pred_test
#predictors = rfe_preds
xgb_reg1.fit(data1[predictors], data1[target])
# use new preds in next submission file, compare label encoding results first
#new_preds = 
#predictors = new_preds
preds = xgb_reg1.predict(data_val[predictors])
#%%
#submission!
submit = pd.DataFrame({"User_ID": test.User_ID, "Product_ID": test.Product_ID, "Purchase": preds})
submit = submit[["User_ID", "Product_ID", "Purchase"]] # need to change order of the columns in the DF!

#%%
# we should only predict positive purchase values 
# Set lowest Purchase value for predicted negative values
print(np.percentile(data1["Purchase"], 0))
lowest_purchase_amount = np.percentile(data1["Purchase"], 0)
#%%
# one solution to change neg. values to min value of train in submit DF.
submit["Purchase"] = np.where(submit["Purchase"]<0,  lowest_purchase_amount, submit["Purchase"])
# with submit.loc or iloc second one?
#submit.loc[submit["Purchase"]< 0,  "Purchase"] = lowest_purchase_amount
#%%
#submit.to_csv('E:/datasets/black friday/xgb_data1_code_nomax_1000trees_notuning_label_encoding_-1test.csv', index=False)
#submit.to_csv('E:/datasets/black friday/xgb_data1_code_maxpurchase_1000trees_notuning_label_encoding_-1test.csv', index=False)
#submit.to_csv('E:/datasets/black friday/xgb_pred_final__maxpurchase_1000trees_notuning_label_encoding_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_pred_test_nomax_1000trees_notuning_label_encoding_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_rfe_preds_nomax_1000trees_notuning_label_encoding_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_rfe_preds_nomax_1500trees_notuning_label_encoding_-1test.csv', index=False)
#submit.to_csv('K:/datasets/black friday/xgb_data1_code_nomax_1500trees_notuning_label_encoding_-1test.csv', index=False)
submit.to_csv('K:/datasets/black friday/xgb__data1_code__nomax__6628trees__tuning_max_depth5___min_child_weight15__label_-1test.csv', index=False)
#%%

# Try out DUMMIES and impact on CV, reduce number of variables with RFE after dummies
#%% 

##########  Pipeline tests ####

#%%
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
class CountPurchasePipe(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None, **fit_params):
        self.columns = columns # groupby and count values that are specified
        
    def fit(self, X, y=None):
        return self # not relevant here
    
    def transform(self, X, y=None):
        output = X.copy(deep=   True)
        if self.columns == None:
            for colname,col in output.iteritems():
                output[colname] = X.groupby(colname)[colname].transform("count") 
            return output
        
        else:
            for colname in self.columns:
                output[colname] = X.groupby(colname)[colname].transform("count") 
            return output[self.columns]
        
#Test

#CountPurchasePipe(["User_ID","Product_ID"]).fit_transform(train)
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
#CustomImp(["Product_Category_2", "Product_Category_3"], value = 0).fit_transform(train)
#%%
#train[train["User_ID"].name] #selects the name from the column and selects that column from the dataframe. Usage: lambda x: data[x.name] ...
#%%
# custom function applied to purchase column - max_purchase
class CustomFunc_max_purchase(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None, value = None, **fit_params):
        self.columns = columns # impute NaN in columns with specified value
        self.value = value
        
    def fit(self, X, y=None):
        return self # not relevant here
    
    def transform(self, X, y=None):
        output = X.copy(deep = True)
        if self.columns == None:
            for colname,col in output.iteritems():
                max_purchase = np.percentile(output[colname], self.value)
                output[output[colname] > max_purchase] = max_purchase
            return output
        
        else:
            for colname in self.columns:
                output[output[colname] > max_purchase] = max_purchase
            return output[self.columns]

#Test
#CustomFunc_max_purchase(["Purchase"], value = 99.9).fit_transform(train)
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
pipe=Pipeline([
        ('sel_cat', ColumnSelector(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])),
        ('enc', LabelEncoderPipe(#columns= "User_ID" # only need to specify columns if we want specific ones. But we specify with ColumnSelector already, so no need.
                ))
])
pipe.fit_transform(train)
"""
#%%
#### or : use pandas and DataFrameMapper:

# train_test_split
mapper = DataFrameMapper([
        (["User_ID"], LabelEncoder()),
        (["Product_ID"], LabelEncoder()),
        (["Gender"], LabelEncoder()),
        (["Age"], LabelEncoder()),
        (["City_Category"], LabelEncoder()),
        (["Occupation"], None),
        (["Marital_Status"], None),
        (["Product_Category_1"], None),
        (["Product_Category_2"], SimpleImputer(strategy="category")),
        (["Product_Category_3"], SimpleImputer(strategy="category")),
        (["Stay_In_Current_City_Years"], LabelEncoder())
      ])
pipe = Pipeline(steps=[
        ('features', mapper),
        ("model", XGBRegressor())
        ])
pipe.fit(X_train,y_train)
pipe.score(X_test,ytest)


#%%

# cross validate imputation of columns cat 2 and 3 and removal
# Loop with cross validate and feature union, preprocessing and pipeline
predictors = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation',
       'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status',
       'Product_Category_1', 'Product_Category_2', 'Product_Category_3']

modes_to_check: ["no_imputation", "imputation_0", "removal"]

CVCompare_columns = ["mode", "RMSE train", "RMSE test", "RMSE test 3*STD"]
CVCompare = pd.DataFrame(columns=CVCompare_columns)
row_index = 0
seed = 42

cachedir = mkdtemp()

for mode in modes_to_check:
    if mode == "imputation_0":
        step = ('pre_nan', Pipeline([ # process our nan columns
                ('sel_cat', ColumnSelector(['Product_Category_2', 'Product_Category_3'])),
                ('cust_imp', CustomImp(value = 0))
                    #('impute', Imputer(strategy = "median")),
                    # should use SimpleImputer and strategy = "category"...? from sklearn.impute
                    # might also influence error: MissingIndicator
                ]))
    
    elif mode == "removal":
        step = ('do_nothing', Pipeline([])) # dont include the columns at all, maybe can test the effect of removing one or the other
    
    else:
        step = ('pre_nan', Pipeline([ # add our nan columns unaltered
                ('sel_cat', ColumnSelector(['Product_Category_2', 'Product_Category_3'])),
                ]))
    
    features = Pipeline([
        ('union', FeatureUnion( 
                # feature union can be part of a pipeline itself
                ## This estimator applies a list of transformer objects in parallel to the input data, then concatenates the results 
                ## This is useful to combine several feature extraction mechanisms into a single transformer

                transformer_list=[ # add list of Pipeline-tuples in transformer_list for concatenation in FeautureUnion
                       
                        #('pre_custom',Pipeline([ # custom feature engineering 
                            #('sel_cust', ColumnSelector(["User_ID"])),
                            #('count_cust', CountPurchasePipe()) # encodes columns to counts. Feature Union should concatenate results
                        #])),
                        
                        ('pre_cat', Pipeline([ # process categorical columns
                            ('sel_cat', ColumnSelector(['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])),
                            ('enc', LabelEncoderPipe()) #columns= "User_ID" ==> only need to specify columns if we want specific ones. But we specify with ColumnSelector already, so no need.
                            #('1hot', OneHotEncoder())
                        ])),
    
                        step,
                
                        ('pre_none', Pipeline([ # add Columns as is
                            ('sel_ordinal', ColumnSelector(['Occupation', 'Product_Category_1']))
                    ])),
                ],
                transformer_weights = None
            )),
        #('select_best', SelectKBest(k=k, score_func = f_regression)), # select k best features - univariate selection. 
        #('standardize', StandardScaler()),
        ('xgb', XGBRegressor(n = 1500, seed = seed, random_state = seed))
        ], memory = cachedir)

    test_pipe = features  

    # splitting strategy defined outside or in here
    #kfold = KFold(n_splits = 5, shuffle = True, random_state=seed)
    #ssplit = ShuffleSplit(n_splits = 5, test_size = 0.05, train_size= 0.1, random_state =seed)
    # evaluate 
    cv_pipe = model_selection.cross_validate(test_pipe, train[predictors], train[target], cv=cv_split, scoring= ("r2", "neg_mean_squared_error"), return_train_score=True)
    
    # store results
    CVCompare.loc[row_index, "mode"] = mode
    CVCompare.loc[row_index, "RMSE train"] = np.sqrt(np.mean(-cv_pipe["train_neg_mean_squared_error"]))
    CVCompare.loc[row_index, "RMSE test"] = np.sqrt(np.mean(-cv_pipe["test_neg_mean_squared_error"]))
    CVCompare.loc[row_index, "RMSE test 3*STD"] = np.sqrt(3*np.std(-cv_pipe["test_neg_mean_squared_error"]))
    row_index+=1
    
    print("results for SKB with mode = {} are: ".format(mode))
    print("rmse train: ", np.sqrt(np.mean(cv_pipe["train_neg_mean_squared_error"])*-1))
    print("rmse test: ", np.sqrt(np.mean(cv_pipe["test_neg_mean_squared_error"])*-1))
    
# print and sort table:
CVCompare.sort_values(by= ["RMSE test"], ascending = True, inplace=True)
print(CVCompare)
print("done with columns: ", cols)

# Clear cache of Pipeline! 
rmtree(cachedir)
  
#%%

####### P C A - Full Data Set #######


# Apply XGB with PCA 

# Pipeline with XGB cross validate 

X = data1[cols]

# should use memory parameter of Pipeline    
cachedir = mkdtemp()
pipe = Pipeline([
     # ('onehot', OneHotEncoder(sparse=True, categorical_features=)),  
     # ('labelencoder', LabelEncoder()), #need to come up with object imputation first
     # ('impute', Imputer()),
     # ('kbest', SelectKBest()),
     # ('minmax', MinMaxScaler(feature_range=(0., 1))),
     ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), # need to scale for PCA 
     ('reduce_dim', PCA(n_components = 82)),
     ('xgb', xgboost.XGBRegressor())
], memory = cachedir)
    
cv_pipe = model_selection.cross_validate(pipe, X, target, cv=cv_split, scoring= "neg_mean_squared_error", return_train_score=True)
print("results are: ")
print("rmse train: ", np.sqrt(np.mean(cv_pipe["train_score"])*-1))
print("rmse test: ", np.sqrt(np.mean(cv_pipe["test_score"])*-1))

# test rmse was 3246 for PCA with 70 comp and cv=cv_split
# test rmse was 3165 for PCA with 82 comp and cv=cv_split


# Clear cache of Pipeline! 
rmtree(cachedir)
#%%

####### S K B - Full Data Set #######


# # Apply SelectKBest to our data now
# Pipeline with XGB cross validate 

X = data1[cols]

# should use memory parameter of Pipeline    
cachedir = mkdtemp()
pipe = Pipeline([
     # ('onehot', OneHotEncoder(sparse=True, categorical_features=)),  
     # ('labelencoder', LabelEncoder()), #need to come up with object imputation first
     # ('impute', Imputer()),
     # ('kbest', SelectKBest()),
     # ('minmax', MinMaxScaler(feature_range=(0., 1))),
     # ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), # need to scale for PCA 
     ('kbest', SelectKBest(k=7, score_func = f_regression)),
     ('xgb', xgboost.XGBRegressor())
], memory = cachedir)
    
cv_pipe = model_selection.cross_validate(pipe, X, target, cv=cv_split, scoring= "neg_mean_squared_error", return_train_score=True)
print("results are: ")
print("rmse train: ", np.sqrt(np.mean(cv_pipe["train_score"])*-1))
print("rmse test: ", np.sqrt(np.mean(cv_pipe["test_score"])*-1))
print("fit time mean: ", cv_pipe["fit_time"].mean())
print("done with columns: ", cols)

# test rmse was 3502 for SelectKBest with k=10 # data1_dummies3 cv=cv_split
# test rmse was 3061 for SelectKBest with k=37 # data1_dummies3 cv=cv_split
# test rmse was 3056 for SelectKBest with k=79 # data1_dummies3 cv=cv_split

# Clear cache of Pipeline! 
rmtree(cachedir)

# Even with only k=10 this takes such a long time to finish. 
# We should look at more k values.
# Maybe reduce dataset for these tests to speed up computation 
# and compare results for small dataset k with full dataset k and see what difference it makes

# far less time consumed!
# test rmse was 3508 for SelectKBest with k=10 # data1_dummies3 cv=cv_split
# test rmse was 3060 for SelectKBest with k=70 # data1_dummies3 cv=cv_split,

# test rmse was 2915 for SelectKBest with k=6 # data1_label cv=ssplit, this is our best result so far

# test rmse was 2925 for SelectKBest with k=6 # data1_label_NaN cv=cv_split
# test rmse was 2923 for SelectKBest with k=3 # data1_label_NaN cv=cv_split
# test rmse was 2922 for SelectKBest with k=2 # data1_label_NaN cv=cv_split, with only two predictors we might generalize better. 

# Which are those? 
# ---> Product ID and Product Category 1!

# k for data1_label_purchase: 3 / 2912 ! new low! count_purchase variable engineered
# ok before we go on with other tests we will build our full model now... model running.
# k for data1_label_purchase_all: 7 / 2908 - new low. We'll run with that now and tune hyperparams of XGB
#%%
SKB =  SelectKBest(k=3, score_func = f_regression)
SKB_fitted = SKB.fit_transform(data1[cols],target)
columns = SKB.get_support()
columns_i = SKB.get_support(indices=True)
for i in columns_i:
    print(cols[i])
# result:
# 'Product_ID' / 'Product_Category_1' / count_purchase are our best 3 features with best RMSE  
# Could we expect to generalize well from that?
    
#%%
####### S K B - Reduced Data Set #######

# test different k values for SelectKBest
# Pipeline with XGB cross validate 

X = data1[cols]
interval = 1
ks = np.arange(1,len(cols), interval)

SelectKBest_columns = ["k", "RMSE train", "RMSE test", "fit time mean"]
SKB_compare = pd.DataFrame(columns=SelectKBest_columns)
row_index = 0
for k in ks:
    
    # should use memory parameter of Pipeline    
    cachedir = mkdtemp()
    pipe = Pipeline([
         ('kbest', SelectKBest(k=k, score_func = f_regression)),
         ('xgb', xgboost.XGBRegressor())
    ], memory = cachedir)
        
    cv_pipe = model_selection.cross_validate(pipe, X, target, cv=ssplit, scoring= "neg_mean_squared_error", return_train_score=True)
    # store results
    SKB_compare.loc[row_index, "k"] = k
    SKB_compare.loc[row_index, "RMSE train"] = np.sqrt(np.mean(cv_pipe["train_score"])*-1)
    SKB_compare.loc[row_index, "RMSE test"] = np.sqrt(np.mean(cv_pipe["test_score"])*-1)
    SKB_compare.loc[row_index, "fit time mean"] = cv_pipe["fit_time"].mean()
    row_index+=1
    print("results for SKB with k = {} are: ".format(k))
    print("rmse train: ", np.sqrt(np.mean(cv_pipe["train_score"])*-1))
    print("rmse test: ", np.sqrt(np.mean(cv_pipe["test_score"])*-1))
    print("fit time mean: ", cv_pipe["fit_time"].mean())
    
# print and sort table:
SKB_compare.sort_values(by= ["RMSE test"], ascending = True, inplace=True)
print(SKB_compare)
print("done with columns: ", cols)

# Clear cache of Pipeline! 
rmtree(cachedir)

# best k for reduced dataset: 79 (3059) 
# almost no increased error and much lower k has 37 (3063)!
# double check the k of 37 and 79  with full dataset and compare to already known error values
# 37: 3061 - marginally better than with the reduced set
# 79: 3056 - marginally better than with the reduced set 

# we can do the same for PCA now
# After that we will change the used variables to include Product_ID and User_ID again. 
# I have a nagging feeling that User_ID or Product_ID might introduce some bias. 
# We will find out with different submissions at the latest.

# Without using StandardScaler we gained computation speed. Results are roughly the same and marginally better:
# best k: 79 (3053); 
# least k with good result: 37 (3059)
# No use in using StandardScaler for SelectKBest then and we drop it. 

# Change the used variables to include Product_ID and User_ID again: # data1_label
# Best k for reduced dataset:  6 (2924) - we have lower error than without IDs and dummied 
# (maybe we can add the IDs and dummy the others only, too?!) 

# # data1_label_NaN
# Best k for reduced dataset: 2 (2928) - ok the columns we dropped in data1_label_NaN didn't improve or worsen prediction! 
# Could have extracted the worst columns out of SKB also
# its interesting that now only 2 columns suffice for a good test RMSE. Compare with full dataset and different ks (2,3,6)

# best k for # data1_dummies3: 37 / 3027

# k for data1_label_purchase: 3 / 2919 ! new low!

# k for data1_label_purchase_all: 7 / 2914 new low
#%%

####### P C A - Reduced Data Set #######

# test different k values for PCA
# Pipeline with XGB cross validate 

X = data1[cols]
interval = 10
ks = np.arange(1,len(cols), interval)

PCA_columns = ["n_components", "RMSE train mean", "RMSE test mean", "fit time mean"]
PCA_compare = pd.DataFrame(columns=PCA_columns)
row_index = 0
for k in ks:
    
    # should use memory parameter of Pipeline    
    cachedir = mkdtemp()
    pipe = Pipeline([
         # ('onehot', OneHotEncoder(sparse=True, categorical_features=)),  
         # ('labelencoder', LabelEncoder()), #need to come up with object imputation first
         # ('impute', Imputer()),
         # ('kbest', SelectKBest()),
         # ('minmax', MinMaxScaler(feature_range=(0., 1))),
         ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
         # need to scale for PCA - we don't use PCA here, does ist change error if we omit this step?
         ('reduce_dim', PCA(n_components=k)),
         ('xgb', xgboost.XGBRegressor())
    ], memory = cachedir)
        
    cv_pipe = model_selection.cross_validate(pipe, X, target, cv=ssplit, scoring= "neg_mean_squared_error", return_train_score=True)
    # store results
    PCA_compare.loc[row_index, "n_components"] = k
    PCA_compare.loc[row_index, "RMSE train mean"] = np.sqrt(np.mean(cv_pipe["train_score"])*-1)
    PCA_compare.loc[row_index, "RMSE test mean"] = np.sqrt(np.mean(cv_pipe["test_score"])*-1)
    PCA_compare.loc[row_index, "fit time mean"] = cv_pipe["fit_time"].mean()
    row_index+=1
    print("results for PCA with k = {} are: ".format(k))
    print("rmse train mean: ", np.sqrt(np.mean(cv_pipe["train_score"])*-1))
    print("rmse test mean: ", np.sqrt(np.mean(cv_pipe["test_score"])*-1))
    print("fit time mean: ", cv_pipe["fit_time"].mean())
    
# print and sort table:
PCA_compare.sort_values(by= ["RMSE test mean"], ascending = True, inplace=True)
print(PCA_compare)
print("done with columns: ", cols)

# Clear cache of Pipeline! 
rmtree(cachedir)

# best k for reduced dataset: 82 (3186) - we have higher error than SKB!
# we can test out the result on the full dataset -> k = 82 results in RMSE of 3165 !
# thus we use SKB for these predictors. Test with other predictors ALSO

# Change the used variables to include Product_ID and User_ID again: 
# best k for reduced dataset: 10 (4405) - we won't try anything further with that. Dropping the NaN columns now!

# best k for # data1_dummies3: 81 / 3230

#%%
####### N M F - Reduced Data Set #######

# Try out if NMF makes a difference
# matrix factorization can capture the interactions between our many levels of feature variables (high cardinality)
# we will give it a try and see if it improves our results

X = data1[cols]
ks = np.arange(1,len(cols),3)

NMF_columns = ["n_components", "RMSE train mean", "RMSE test mean", "fit time mean"]
NMF_compare = pd.DataFrame(columns=NMF_columns)
row_index = 0
for k in ks:
    
    # should use memory parameter of Pipeline    
    cachedir = mkdtemp()
    pipe = Pipeline([
         # ('onehot', OneHotEncoder(sparse=True, categorical_features=)),  
         # ('labelencoder', LabelEncoder()), #need to come up with object imputation first
         # ('impute', Imputer()),
         # ('kbest', SelectKBest()),
         # ('minmax', MinMaxScaler(feature_range=(0., 1))),
         ('scaler', StandardScaler(copy=True, with_mean=False, with_std=True)), # need that for NMF?
         # maybe use StandardScaler for NMF later with "with_mean" = False (no negative values) and compare RMSE
         ('reduce_dim', NMF(n_components=k)),
         ('xgb', xgboost.XGBRegressor())
    ], memory = cachedir)
        
    cv_pipe = model_selection.cross_validate(pipe, X, target, cv=ssplit, scoring= "neg_mean_squared_error", return_train_score=True)
    # store results
    NMF_compare.loc[row_index, "n_components"] = k
    NMF_compare.loc[row_index, "RMSE train mean"] = np.sqrt(np.mean(cv_pipe["train_score"])*-1)
    NMF_compare.loc[row_index, "RMSE test mean"] = np.sqrt(np.mean(cv_pipe["test_score"])*-1)
    NMF_compare.loc[row_index, "fit time mean"] = cv_pipe["fit_time"].mean()
    row_index+=1
    print("results for NMF with k = {} are: ".format(k))
    print("rmse train mean: ", np.sqrt(np.mean(cv_pipe["train_score"])*-1))
    print("rmse test mean: ", np.sqrt(np.mean(cv_pipe["test_score"])*-1))
    print("fit time mean: ", cv_pipe["fit_time"].mean())
    
# print and sort table:
NMF_compare.sort_values(by= ["RMSE test mean"], ascending = True, inplace=True)
print(NMF_compare)
print("done with columns: ", cols)

# Clear cache of Pipeline! 
rmtree(cachedir)

# Fit times are generally higher for NMF than for the other two dim. red. techniques
# Best k for reduced dataset: 64 (3032) - we have lower error than SKB! # no StandardScaler
# Already, NMF results are better than PCA or SKB. Let's see if scaling improves RMSE further.

# With StandardScaler (mean = False):
# Best k for reduced dataset: 73 (3057) -  # ... too big k steps
# ==> Scaling improves ?
# we can test out the result on the full dataset -> k = results in RMSE of !

# we can test out the result on the full dataset -> k = results in RMSE of !
# Test with other predictors ALSO

# best k for reduced dataset: 82 (3186) - we have higher error than SKB! # with Scaler (mean = False)
# we can test out the result on the full dataset -> k = results in RMSE of !
# Test with other predictors ALSO

# Change the used variables to include Product_ID and User_ID again: # data1_label
# With StandardScaler (mean = False):
# Best k for reduced dataset:  10 (3588) - we have higher error than without IDs and dummied (maybe we can add the IDs and dummy the others only)! 
# Without Scaler:
# Best k for reduced dataset:  8 (4223) use scaler!

# data1_dummies3:
# best k results in higher error than SKB with other column selection. Fit time is vast.
#%%
####### N M F - Full #######

# Try out if NMF makes a difference
# matrix factorization can capture the interactions between our many levels of feature variables (high cardinality)
# we will give it a try and see if it improves our results

X = data1[cols]
k = 55


# should use memory parameter of Pipeline    
cachedir = mkdtemp()
pipe = Pipeline([
     # ('onehot', OneHotEncoder(sparse=True, categorical_features=)),  
     # ('labelencoder', LabelEncoder()), #need to come up with object imputation first
     # ('impute', Imputer()),
     # ('kbest', SelectKBest()),
     # ('minmax', MinMaxScaler(feature_range=(0., 1))),
     ('scaler', StandardScaler(copy=True, with_mean=False, with_std=True)), # need that for NMF?
     # maybe use StandardScaler for NMF later with "with_mean" = False (no negative values) and compare RMSE
     ('reduce_dim', NMF(n_components=k)),
     ('xgb', xgboost.XGBRegressor())
], memory = cachedir)
    
cv_pipe = model_selection.cross_validate(pipe, X, target, cv=cv_split, scoring= "neg_mean_squared_error", return_train_score=True)
print("results for NMF with k = {} are: ".format(k))
print("rmse train mean: ", np.sqrt(np.mean(cv_pipe["train_score"])*-1))
print("rmse test mean: ", np.sqrt(np.mean(cv_pipe["test_score"])*-1))
print("fit time mean: ", cv_pipe["fit_time"].mean())
    

# Clear cache of Pipeline! 
rmtree(cachedir)

# data1_dummies3:
# best k = 55: 
#%%
# ideas for feature engineering - could introduce a feature like :
# - how many purchases did the user make in total? Single or multiple?
# - how many ... no nothing comes to mind yet :)

# could also try dropping the category 2 and 3 entirely
# could ensemble a few XGBs model predictions with different seeds after we tuned XGB hyperparameters

#%%

