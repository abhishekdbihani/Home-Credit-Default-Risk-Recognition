## <b> Capstone Project - Machine Learning Engineer Nanodegree - Udacity </b>

## <b>  Home Credit Default Risk Recognition </b>

### <b> -By Abhishek Bihani </b>

### Based on the [Kaggle Competition]((https://www.kaggle.com/c/home-credit-default-risk/overview))

---

<b> Domain Background </b>

An important fraction of the population finds it difficult to get their home loans approved due to insufficient or absent credit history. This prevents them to buy their own dream homes and at times even forces them to rely on other sources of money which may be unreliable and have exorbitant interest rates. Conversely, it is a major challenge for banks and other finance lending agencies to decide for which candidates to approve housing loans. The credit history is not always a sufficient tool for decisions, since it is possible that those borrowers with a long credit history can still default on the loan and some people with a good chance of loan repayment may simply not have a sufficiently long credit history. 

A number of recent researchers have applied machine learning to predict the loan default risk. This is important since a machine learning-based classification tool to predict the loan default risk which uses more features than just the traditional credit history can be of great help for both, potential borrowers, and the lending institutions. 

---

<b> Problem Statement </b>

The [problem and associated data](https://www.kaggle.com/c/home-credit-default-risk/overview) has been provided by Home Call Credit Group for a Kaggle competition. The problem can be described as, <i> “A binary classification problem where the inputs are various features describing the financial and behavioral history of the loan applicants, in order to predict whether the loan will be repaid or defaulted.” </i> 

---

<b> Project Novelty </b>

<i> The [notebook](https://github.com/abhishekdbihani/Home-Credit-Default-Risk-Recognition/blob/master/Abhishek%20Capstone%20-%20Home%20Credit%20Risk%20v2.ipynb) provides a complete end-to-end machine learning workflow for building a binary classifier, and includes methods like automated feature engineering for connecting relational databases, comparison of different classifiers on imbalanced data, and hyperparameter tuning using Bayesian optimization. </i>

<img src="https://github.com/abhishekdbihani/Home-Credit-Default-Risk-Recognition/blob/master/roc_auc_compare.PNG" align="middle" width="500" height="400" alt="ROC AUC comparison" >
Figure 1- ROC Curve and AUC Comparison of Different Classifiers

---

<b> Datasets and Inputs </b>

The [dataset files](https://www.kaggle.com/c/home-credit-default-risk/data) are provided on the Kaggle website in the form of multiple CSV files and are free to download. The dataset files are described as per Figure 2.

![image](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)
 
Figure 2- Description and connectivity of the Home Credit Default Risk dataset

As seen in Figure 2, the file application_{train|test}.csv contains the main table containing the training dataset (307511 samples) and test dataset (48744 samples), with each row representing one loan identified by the feature SK_ID_CURR. The training set contains the variable TARGET with binary values (0: the loan was repaid or 1: the loan was not repaid). There are many input files available, which can be analysed for input features to train the model. The large number of input features and training samples make it easier to identify the important factors and for constructing a credit default risk classification model.

---

<b> Project Design and Solution </b>

The [project](https://github.com/abhishekdbihani/Home-Credit-Default-Risk-Recognition/blob/master/Abhishek%20Capstone%20-%20Home%20Credit%20Risk%20v2.ipynb) has been divided into five parts-

1. <u>Data Preparation</u> - Before starting the modeling, we need to import the necessary libraries and the datasets. If there are more than one files, then all need to be imported before we can look at the feature types and number of rows/columns in each file. 

2. <u>Exploratory Data Analysis</u> - After data importing, we can investigate the data and answer questions like- How many features are present and how are they interlinked? What is the data quality, are there missing values? What are the different data types, are there many categorical features? Is the data imbalanced? And most importantly, are there any obvious patterns between the predictor and response features? 

3. <u>Feature Engineering</u> - After exploring the data distributions, we can conduct feature engineering to prepare the data for model training. This includes operations like replacing outliers, imputing missing values, one-hot encoding categorical variables, and rescaling the data. Since there are number of relational databases, we can use extract, transform, load (ETL) processes using automated feature Engineering with [Featuretools](https://www.featuretools.com/) to connect the datasets. The additional features from these datasets will help improve the results over the base case (logistic regression). 

4. <u>Classifier Models: Training, Prediction and Comparison</u> - After the dataset is split into training and testing sets, we can correct the data imbalances by undersampling the majority class. Then, we can training the different classifier models (Logistic Regression, Random Forest, Decision Tree, Gaussian Naive Bayes, XGBoost, Gradient Boosting, LightGBM) and compare their performance on the test data using metrics like accuracy, F1-score and ROC AUC. After choosing the best classifier, we can use K-fold cross validation to select the best model. This will help us choose parameters that correspond to the best performance without creating a separate validation dataset.

5. <u>Hyperparameter Tuning</u> - After choosing the binary classifier, we can tune the hyperparameters for improving the model results through grid search, random search, and Bayesian optimization [(Hypertopt library)](https://github.com/hyperopt/hyperopt). The hyperparameter tuning process will use an objective function on the given domain space, and an optimization algorithm to give the results. The ROC AUC validation scores from all three methods for different iterations can be compared to see trends.

---

<b> Package/Library Requirements </b>

The following packages need to be installed for running the [project notebook](https://github.com/abhishekdbihani/Home-Credit-Default-Risk-Recognition/blob/master/Abhishek%20Capstone%20-%20Home%20Credit%20Risk%20v2.ipynb).

1) sklearn  - For models and metrics<br>
2) warnings - For preventing warnings<br>
3) numpy - For basic matrix handling<br>
4) matplotlib - For figure plotting<br>
5) pandas - For creating dataframes<br>
6) seaborn - For figure plotting<br>
7) timeit - For tracking times<br>
8) os - for setting work directory<br>
9) random - For creating random seeds<br>
10) csv - For saving csv files<br>
11) json - For creating json files<br>
12) itertools - For creating iterators for efficient looping<br>
13) pprint - For pretty printing data structures<br>
14) pydash - for doing “stuff” in a functional way (utility library). <br>
15) gc -  Garbage collector for deleting data <br>
16) re - Raw string notation for regular expression patterns <br>
17) featuretools - Automated feature engineering <br>
18) xgboost - XGBoost model <br>
19) lightgbm - LightGBM model <br>
20) hyperopt - Bayesian hyperparameter optimization <br>

<i> Note - The packages can be installed by uncommenting the first cell in the [project notebook](https://github.com/abhishekdbihani/Home-Credit-Default-Risk-Recognition/blob/master/Abhishek%20Capstone%20-%20Home%20Credit%20Risk%20v2.ipynb) </i> 

<b> References / Acknowledgements </b>

This project builds on scripts and explanations from other Jupyter notebooks publicly shared on Kaggle. The list is as follows-

1) [A Gentle Introduction](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction) <br>
2) [Introduction to Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics) <br>
3) [Advanced Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory) <br>
4) [Intro to Model Tuning: Grid and Random Search](https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search) <br>
5) [Automated Model Tuning](https://www.kaggle.com/willkoehrsen/automated-model-tuning) <br>
6) [Home Credit Default Risk Extensive EDA](https://www.kaggle.com/gpreda/home-credit-default-risk-extensive-eda) <br>
7) [Home Credit Default Risk: Visualization & Analysis](https://www.kaggle.com/charlievbc/home-credit-default-risk-visualization-analysis) <br>
8) [Loan repayers v/s Loan defaulters - HOME CREDIT](https://www.kaggle.com/pavanraj159/loan-repayers-v-s-loan-defaulters-home-credit) <br>

---
