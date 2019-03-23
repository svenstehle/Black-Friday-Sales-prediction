# AnalyticsVidhya's Black Friday Data Hack
[Practice-Problem-Black-Friday](https://datahack.analyticsvidhya.com/contest/black-friday/)

name on Leaderboard: 	**intradeep**

### Online data science hackathon originally held Nov 19-22 2015 by Analytics Vidhya. 
Now it's a practice problem. 

### Problem Statement:
* A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.
* The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
* Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

### Approach:
* I spent some time doing EDA, see the code file for that. Commented that part out.
* Invested some time encoding the variable columns the right way. 
* Very important was to mark unknown samples in a way that the algorithm could notice that (new product ids in test data).
* After that I looked at possible algorithms for modeling and decided on XGB based on initial cv results.

* I compared RMSE of the base variable set with that of an interaction term dataset. I compared a lot of feature sub sets results - this includes select k best as well as selection based on xgb feature importance and pca.
* Based on RMSE CV results I decided on the base varible set...
* ...and proceeded to tune the hyperparameters of this model. This model was used as the final model and I ensembled 5 tuned XGBs with different seeds (averaging) for the result of RMSE 2467. 

* I also tried out stacking different algorithms but to no avail.
* GLM Regression techniques fared pretty good (~2900 RMSE) when I changed the dataset to dummies but was horrible for the base dataset.

### Possible improvements:
* Employ a neural network
* Explore GLM more and possibly eliminate some (dummy) variables 

### Thoughts:
All in all this dataset was a huge challenge for me because of the sheer size and time it took to try out different approaches and compute results. I had a lot of fun and learned a lot about interaction terms, even though they were not very useful in this case, and xgb parameters and tuning.
I have a lot of respect for the ladies and gents that managed to achieve great RMSE results in the initial hackathon during just a few hours. I am curious to see what the best teams did and to learn from their approaches.
