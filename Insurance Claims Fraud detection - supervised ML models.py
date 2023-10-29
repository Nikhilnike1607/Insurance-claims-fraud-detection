#!/usr/bin/env python
# coding: utf-8

# # Project : Insurance Claims Fraud Detection - Supervised ML
# 
# 
# ## Business Scenario: 
# 
# <font color= Dark Green> An insurance company aims to reduce financial losses due to fraudulent claims by implementing an advanced fraud detection system. The company has a dataset containing historical insurance claims and wants to build a machine learning model to predict and identify potentially fraudulent claims, thereby saving costs and improving the integrity of the insurance process. </font>
# 
# 
# ### Project/Colab Overview
# >  Import Libraries </br>
# >  Insights from our Dataset </br>
# >  Analysis on unique values in each column </br>
# >  Dealing with missing values and outliers </br>
# >  Feature engineering </br>
# >  categorical and numerical columns </br>
# >  Train and Test split </br>
# 
# ### Supervised ML Models are used to compare:
# > Support Vector classifier </br>
# > Decision Tree classifier </br>
# > Random Forest classifier </br>
# > Adaboost classifier </br>
# > Gradient boosting </br>
# > Stochastic Gradient Boosting </br>
# > Extra Trees Classifier </br>
# > VotingClassifier </br>

# ### Import Libraries

# In[173]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import warnings
warnings.filterwarnings('ignore')
sns.set_style('dark')

plt.style.use('ggplot')
pd.set_option('display.max_columns', None)


# In[174]:


#Improt the dataset and first look of the dataset

data = pd.read_csv('Insurance_claims_Fradulent_detection.csv')
data.head()


# In[175]:


data.info()


# In[176]:


data.shape


# In[177]:


data.columns


# ### Exploratory Data analysis

# In[178]:


##Checking null values

(data.isnull().sum()/data.shape[0])*100


# In[179]:


#Dropping columns with above 60% missing values
data.drop(columns = '_c39', axis = 1, inplace= True)


# In[180]:


data.head()


# In[181]:


# Replacing values with '?' with nan values
data.replace('?', np.nan, inplace = True)
data.head()


# In[182]:


data[data['authorities_contacted'].isnull()]


# In[183]:


data['authorities_contacted'].value_counts()


# In[184]:


data['police_report_available'].value_counts()


# In[185]:


data['collision_type'].value_counts()


# In[186]:


data['property_damage'].value_counts()


# In[187]:


data['police_report_available'].unique()


# In[188]:


#statiscal overview of numerical data in the dataset
data.describe()


# In[189]:


# checking the percentage of fraus reported
fraud_reported_count = data['fraud_reported'].value_counts()
fraud_reported = (fraud_reported_count/len(data))*100

plt.figure(figsize = (3,3))
plt.pie(fraud_reported, labels=fraud_reported_count.index, autopct='%1.1f%%')
plt.title('Frauds reported distribution')
plt.show()


# In[190]:


gender_count = data['insured_sex'].value_counts()
gender = (fraud_reported_count/len(data))*100

plt.figure(figsize = (3,3))
plt.pie(fraud_reported, labels=fraud_reported_count.index, autopct='%1.1f%%')
plt.title('Frauds reported distribution based on insured gender')
plt.show()


# In[191]:


data['collision_type'].value_counts()


# In[192]:


# imputing null values using mode
data['collision_type'] = data['collision_type'].fillna(data['collision_type'].mode()[0])


# In[193]:


data.head()


# In[194]:


data.isnull().sum()


# In[195]:


#missing values
cols_with_na = [cols for cols in data.columns if data[cols].isnull().sum()>1]


# In[196]:


for col in cols_with_na:
    print(col, np.round(data[col].isnull().mean(), 4), '% of missing values')


# In[197]:


#see if missing value columns have any relationship with dependent variable
for col in cols_with_na:
    data1 = data.copy()
    
    data1[col]=np.where(data[col].isnull(),1,0)
    data1.groupby(col)['total_claim_amount'].median().plot.bar()
    plt.title(col)
    plt.show()


# In[198]:


#imputing the missing data
data['collision_type'] = data['collision_type'].fillna(data['collision_type'].mode()[0])
data['authorities_contacted'] = data['authorities_contacted'].fillna(data['authorities_contacted'].mode()[0])
data['property_damage'] = data['property_damage'].fillna(data['property_damage'].mode()[0])
data['police_report_available'] = data['police_report_available'].fillna(data['police_report_available'].mode()[0])


# In[199]:


data.isna().sum()


# In[200]:


data.nunique()


# In[201]:


# dropping columns which are not necessary for prediction

to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year','policy_csl']

data.drop(to_drop, inplace = True, axis = 1)


# In[202]:


data.head()


# In[203]:


data.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)


# In[204]:


data.head()


# In[205]:


X = data.drop('fraud_reported', axis = 1)
y = data['fraud_reported']


# In[206]:


# extracting categorical columns
cat_df = X.select_dtypes(include = ['object'])


# In[207]:


cat_df.head()


# In[208]:


# printing unique values of each column
for col in cat_df.columns:
    print(f"{col}: \n{cat_df[col].unique()}\n")


# In[209]:


#using get_dummies for categorical variables
cat_df = pd.get_dummies(cat_df, drop_first = True)
cat_df = cat_df.astype(int)


# In[210]:


cat_df.head()


# In[211]:


# extracting the numerical columns
num_df = X.select_dtypes(include = ['int64'])
num_df.head()


# In[212]:


# checking for multicollinearity

plt.figure(figsize = (18, 12))

corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
plt.show()


# In[213]:


# combining the Numerical and Categorical dataframes to get the final dataset

X = pd.concat([num_df, cat_df], axis = 1)
X.head()


# In[214]:


plt.figure(figsize = (25, 20))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(X[col])
        plt.xlabel(col, fontsize = 15)
        
    plotnumber += 1
    
plt.tight_layout()
plt.show()


# Outliers are present in some numerical columns we will scale numerical columns later

# In[215]:


# splitting data into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[216]:


X_train.head()


# In[217]:


num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]


# In[218]:


# Scaling the numeric values in the dataset

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)


# In[219]:


scaled_num_df = pd.DataFrame(data = scaled_data, columns = num_df.columns, index = X_train.index)
scaled_num_df.head()


# In[220]:


X_train.drop(columns = scaled_num_df.columns, inplace = True)


# In[221]:


X_train = pd.concat([scaled_num_df, X_train], axis = 1)


# In[222]:


X_train.head()


# ## Model : Support Vector classifier

# In[223]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)


# In[224]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
svc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ## Decision Tree classifier model

# In[225]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)


# In[226]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[227]:


# hyper parameter tuning

from sklearn.model_selection import GridSearchCV

grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)


# In[228]:


# best parameters and best score

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[229]:


# best estimator 

dtc = grid_search.best_estimator_

y_pred = dtc.predict(X_test)


# In[230]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ## Model : Random Forest classifier

# In[231]:


from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)


# In[232]:


# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ## Model : Adaboost classifier

# In[233]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(base_estimator = dtc)

parameters = {
    'n_estimators' : [50, 70, 90, 120, 180, 200],
    'learning_rate' : [0.001, 0.01, 0.1, 1, 10],
    'algorithm' : ['SAMME', 'SAMME.R']
}

grid_search = GridSearchCV(ada, parameters, n_jobs = -1, cv = 5, verbose = 1)
grid_search.fit(X_train, y_train)


# In[234]:


# best parameter and best score

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[235]:


# best estimator 

ada = grid_search.best_estimator_

y_pred = ada.predict(X_test)


# In[236]:


# accuracy_score, confusion_matrix and classification_report

ada_train_acc = accuracy_score(y_train, ada.predict(X_train))
ada_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Ada Boost is : {ada_train_acc}")
print(f"Test accuracy of Ada Boost is : {ada_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ## Model: Gradient boosting

# In[237]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of gradient boosting classifier

gb_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")


# ## Model : Stochastic Gradient Boosting (SGB)

# In[238]:


sgb = GradientBoostingClassifier(subsample = 0.90, max_features = 0.70)
sgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier

sgb_acc = accuracy_score(y_test, sgb.predict(X_test))

print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")


# ## Model : Extra Trees Classifier

# In[239]:


from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of extra trees classifier

etc_acc = accuracy_score(y_test, etc.predict(X_test))

print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")


# In[240]:


from sklearn.ensemble import VotingClassifier

classifiers = [('Support Vector Classifier', svc),  ('Decision Tree', dtc), ('Random Forest', rand_clf),
               ('Ada Boost', ada), ('Gradient Boosting Classifier', gb), ('SGB', sgb),
              ('Extra Trees Classifier', etc)]

vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)


# In[241]:


# accuracy_score, confusion_matrix and classification_report

vc_train_acc = accuracy_score(y_train, vc.predict(X_train))
vc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Voting Classifier is : {vc_train_acc}")
print(f"Test accuracy of Voting Classifier is : {vc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[242]:


#creating dataframe for models and obtained scores
models = pd.DataFrame({
    'Model' : ['SVC', 'Decision Tree', 'Random Forest','Ada Boost', 'Gradient Boost', 'SGB', 'Extra Trees', 'Voting Classifier'],
    'Score' : [svc_test_acc, dtc_test_acc, rand_clf_test_acc, ada_test_acc, gb_acc, sgb_acc, etc_acc, vc_test_acc]
})


models.sort_values(by = 'Score', ascending = False)


# ## Model Comparisons:
#     
# Using bar plot, comparing the obtained test accuracies of ML models used

# In[243]:


px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Comparison of Models')

