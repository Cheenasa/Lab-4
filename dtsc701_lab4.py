#!/usr/bin/env python
# coding: utf-8

# KNN
# 
# In this lab, you are going to practice data preprocessing and building the KNN model using
# MLLib and other spark tools.
# Go to the following website and download the dataset. Training data is already given to you as
# train.csv. Your goal is to build a model that can accurately predict survival in test.csv.
# https://www.kaggle.com/competitions/titanic/overview
# 
# Part 1 - Build a KNN classifier to classify the dataset.
# * ● Write standard scaler from scratch - do not scale/z-score features using off-the-shelf scaler from sklearn
# * ● Scale the data(where appropriate) using standard scaler
# * ● Split the dataset into training and testing
# * ● Determine the K value, and create a visualization of the accuracy. Report the best Kvalue
# * ● Run 5 fold cross validations - report mean and standard deviation
# * ● Evaluate using confusion matrix
# * ● Use MARKDOWN cell to explain the accuracy of your model
# 

# ### Data Dictionary
# 
# Variable	Definition	          Key
# survival	Survival	          0 = No, 1 = Yes
# pclass	    Ticket class	      1 = 1st, 2 = 2nd, 3 = 3rd
# sex	        Sex	
# Age	        Age in years	
# sibsp	    # of siblings / spouses aboard the Titanic	
# parch	    # of parents / children aboard the Titanic	
# ticket	    Ticket number	
# fare	    Passenger fare	
# cabin	    Cabin number	
# embarked	Port of Embarkation	  C = Cherbourg, Q = Queenstown, S = Southampton
# 
# 
# #### Variable Notes
# 
# * pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# * age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[1]:


#Importing relevant libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.float_format", lambda x: "%.3f" % x)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


#importing dataset
data = pd.read_csv('/Users/cheerycheena/Downloads/train.csv')


# In[3]:


data.head(10)


# In[4]:


#Checking for duplicate record on train set
data.duplicated().sum()


# In[5]:


#Checking for missing values
pd.DataFrame(data={'% of Missing Values':round(data.isna().sum()/data.isna().count()*100,2)})


# * Missing values are found in the Age, Cabin and embarked attributes. The percentages are quite high.
# * Age has 19.87% missing values out of the total observations.
# * Cabin has 77.10% missing values out of the total observations.

# ### Making sense of the data (Exploratory Data Analysis)

# In[6]:


data.info()


# * There are missing data in the dataset.
# * There are 891 records and 12 attributes.
# * Name, Passenger ID and Ticket no do not provide any predictive value and are therefore not necesary for our model. They should be dropped.

# In[7]:


data.nunique()


# * This confirms that PassengerID and Name should be dropped because they will add no value to the model.

# In[8]:


#Getting the number of people that survived in each socio-economic status (pclass)
data.groupby('Pclass')['Survived'].sum()


# * The upper class survived the most with 136 observations.
# * The middle class survived the least with 87 observations.

# In[9]:


data.describe().T


# * Survived is the output label and should be a categorical variable.
# * Max age in the dataset is 80. the mean and median is 28 and 29.7 respectively. The fractions in the age represents estimated ages and records of people less than one year old.
# * There are outliers in the Fare attribute.

# In[10]:


data.describe(include='object').T


# * Males are the majority with a count of 577
# * Port of embarkment S (Southampton) has the highest record of 644.

# In[11]:


data.hist(figsize = (14,14));
plt.show()


# The dataset has:
# * A record of more people who did not survive than those who did.
# * Lower class (pclass=3) is most frequent.
# * Most of the people are in the age range of 19-22years.
# * Most Fare fell within the range of 0 to 150.

# In[12]:


sns.heatmap(data=data.corr(), annot=True);


# From the heatmap:
# * Fair and survived are more correlated to each other than other attributes. 
# * Same goes for Parch and Sibsp.
# 

# In[13]:


sns.pairplot(data=data, hue='Survived');


# * Those with lower age and higher fare in pclass of 1 survived greatly.

# In[14]:


#Making a copy of the dataset to avoid changes to the original dataset.
df = data.copy()
df.tail(5)


# In[15]:


#dropping PassengerID, Name and Cabin and Ticket attributes.

df.drop(columns = ['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True)


# ### Dealing with missing values
# 
# The Age and embarked attribute has missing values. We will use simple imputer to affix the missing values. 

# In[16]:


df.isna().sum()


# In[17]:


#For Age, we will replace the missing values with the median value.
median_value = df['Age'].median()
df["Age"].fillna(value = median_value, inplace=True)


# In[18]:


#For Embarked, missing values constitute of only 2 observations. We can drop the rows as this will not have a 
#significant effect on dataset.
df=df.dropna()


# In[19]:


df.isna().sum()


# * All missing values have been dealt with.

# ### One Hot Encoding.
# 
# One hot encoding converts categorical data needs to be represented in numerical format  to enable us build the model. The Sex and Embarked column are the two columns that needs to be one-hot encoded.

# In[20]:


df1 = pd.get_dummies(df, columns=['Sex', 'Embarked'])
df1


# In[21]:


#dropping the Sex_female column. It's a redundant attribute because Sex_male attribute captures the same information.

df1 = df1.drop(columns='Sex_female')
df1


# ### Building the Model
# 
# The train and test data set has been given differently, therefore there in no need to split the dataset.
# 
# For the train dataset:
# 

# In[22]:


X = df1.drop(columns='Survived', axis = 1)
Y = df1['Survived']


# ### Standardization of the dataset
# 
# Most of the attributes have values btw 0 and 1. Therefore we will only standardize the age and fare attributes.

# In[23]:


# defining a function to standardize the dataset
def standardize_dataset(df):
    # Calculate the mean and standard deviation of each attribute
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    
    # Standardize each attribute
    for col in df.columns:
        df[col] = (df[col] - means[col]) / stds[col]
    return df


# In[24]:


# Standardize the dataset
X = standardize_dataset(X)
X


# ### Splitting the dataset

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1,stratify=Y)
print(X_train.shape, X_test.shape)


# ### Feature engineering for the Test Dataset.
# 
# This includes dealing with missing values, dropping unnecessary columns, normalization and hot encoding.

# In[26]:


data_test = pd.read_csv('/Users/cheerycheena/Downloads/test.csv')


# In[27]:


data_test.head(10)


# In[28]:


data_test.duplicated().sum()


# In[29]:


data_test.info()


# * There are issing values in the Age, Fare and Cabin columns.
# * The missing values in the age column will be filled with median value.
# * We need to drop the Cabin column and the 2 null rows in the Fare columns.
# * The PassengerId, Ticket and Name column which do not have any predictive value, should also be dropped.

# In[30]:


#For Age, we will replace the missing values with the median value.
median_value = data_test['Age'].median()
data_test["Age"].fillna(value = median_value, inplace=True)

#Dropping the Cabin column
data_test.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)

#Dropping the rows with missing values
data_test = data_test.dropna()
data_test.info()


# * the test data set has 417 observations and 6 columns

# In[31]:


#One-Hot-Encoding for test dataset
data_test_encoded = pd.get_dummies(data_test.copy(), columns=['Sex', 'Embarked'])

#Standardardization for the test dataset
data_test_stand = standardize_dataset(data_test_encoded.copy())
data_test_stand


# In[32]:


x = data_test_stand.drop(columns='Sex_female')


# In[33]:


X.shape


# In[34]:


x.shape


# ### Determining the value of neighbours(k)
# 
# There are various methods to determining the optimal value of k. This includes elbow method, silhoutte coefficient, gap statistics, etc.
# 
# #### The Elbow Method:
# 
# This Computes the within-cluster sum of squares (WCSS) for different values of k (the number of clusters) and plots the results.
# 

# In[35]:


def elbow_method(X, k_range):
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
    plt.plot(k_range, wcss)
    plt.xlabel('k (number of clusters)')
    plt.ylabel('WCSS (within-cluster sum of squares)')
    plt.title('Elbow Method')
    plt.show()


# In[36]:


from sklearn.cluster import KMeans


# In[37]:


k_range = range(1, 50)
elbow_method(X, k_range)


# * We can see from the graph that 9,15 and 24 will make good number of clusters. We will build models with the different number of clusters and compare their accuracies.

# ### KNN for 9 clusters

# In[41]:


knn_9 = KNeighborsClassifier(n_neighbors=9)
knn_9.fit(X_train, y_train)


# In[43]:


y_pred = knn_9.predict(X_test)


# In[45]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[48]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# ### KNN for 15 clusters

# In[53]:


knn_15 = KNeighborsClassifier(n_neighbors=15)
knn_15.fit(X_train, y_train)


# In[54]:


y_pred_15 = knn_15.predict(X_test)


# In[55]:


accuracy = accuracy_score(y_test, y_pred_15)
print('Accuracy:', accuracy)


# In[56]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_15)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# In[57]:


knn_24 = KNeighborsClassifier(n_neighbors=24)
knn_24.fit(X_train, y_train)


# In[58]:


y_pred_24 = knn_24.predict(X_test)


# In[60]:


accuracy = accuracy_score(y_test, y_pred_24)
print('Accuracy:', accuracy)


# In[59]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_24)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# At k = 24:
# 
# The KNN classifier model has the highest accuracy with an accuracy of 82%. 
# 
# 

# In[ ]:




