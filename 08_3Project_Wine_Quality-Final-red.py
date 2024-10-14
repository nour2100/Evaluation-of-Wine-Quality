#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# In[2]:


wine=pd.read_csv("winequality-red.csv",sep=';')


# In[3]:


wine.head()


# In[4]:


wine.shape


# In[5]:


wine.info()


# In[6]:



wine.describe()


# In[7]:


# to find null values
wine.isna().sum()


# In[8]:


# drop null values
wine.dropna(inplace=True)


# In[9]:


wine.drop_duplicates(inplace=True)


# In[10]:


#wine.drop(columns="type",inplace=True)


# In[11]:


wine.shape


# In[12]:


correlation=wine.corr()


# Pearson correlation: The Pearson correlation is the most commonly used measurement for a linear relationship between two variables. The stronger the correlation between these two datasets, the closer it'll be to +1 or -1.

# In[13]:



sns.set(rc={"figure.figsize":(12, 12)}) #width=3, #height=4
sns.heatmap(correlation,annot=True)


# # Observation :- total sulfur dioxide and free sulfur dioxide are positive correlated

# In[14]:


wine.quality.unique()


# In[15]:


wine.quality.value_counts() # Quality :- 0,1,2 and 10 not present


# In[16]:


plt.bar(wine.quality.unique(),wine.quality.value_counts())


# # Replacement of quality score with low,medium, high

# group_Qulaity={3:"low" , 4: "low" , 5: "low" , 6: "medium" ,7: "high" , 8: "high" , 9: "high" }
# wine["quality"]=wine["quality"].replace(group_Qulaity)

# In[17]:


bins = (2,6.5,9)
group_names = ['low','high']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[18]:


plt.bar(wine.quality.unique(),wine.quality.value_counts())


# In[19]:


wine.quality.value_counts()


# In[20]:


# For wine Quality - Label encoding
# label encode the target variable
wine.quality = LabelEncoder().fit_transform(wine.quality)


# In[21]:


wine.quality.unique()


# In[22]:


# Input
X=wine.drop("quality",axis=1)

# Output
y=wine.quality


# In[ ]:





# # Encoding for Wine Type input
# w_type={"white":1,"red":0}
# X["type"]=X["type"].replace(to_replace=w_type)

# # Dealing with Outliers
# fixed acidity	residual sugar, free sulfur dioxide	total sulfur dioxide	

# In[23]:


# Before handling Outliers
plt.boxplot(X["fixed acidity"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Fixed acidity')


# In[24]:


X["fixed acidity"].quantile(0.25),X["fixed acidity"].quantile(0.75)
IQR=X["fixed acidity"].quantile(0.75)-X["fixed acidity"].quantile(0.25)
print("IQR :",IQR)

# whisker values
Q1=X["fixed acidity"].quantile(0.25)
Q3=X["fixed acidity"].quantile(0.75)
print("Q1 and Q3 : ",Q1,Q3)
whisker_value1=Q1-(1.5*IQR)
whisker_value2=Q3+(1.5*IQR)

print("whisker 1 and whisker 2 :",whisker_value1,whisker_value2)

X["fixed acidity"].loc[X["fixed acidity"]<3.95]=Q1

X["fixed acidity"].loc[X["fixed acidity"]>12.3]=Q3


# In[25]:


# After handling Outliers
plt.boxplot(X["fixed acidity"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Fixed acidity')


# In[26]:


# Before handling Outliers
plt.boxplot(X["residual sugar"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Residual Sugar')


# In[27]:


X["residual sugar"].quantile(0.25),X["residual sugar"].quantile(0.75)
IQR=X["residual sugar"].quantile(0.75)-X["residual sugar"].quantile(0.25)
print("IQR :",IQR)

# whisker values
Q1=X["residual sugar"].quantile(0.25)
Q3=X["residual sugar"].quantile(0.75)
print("Q1 and Q3 : ",Q1,Q3)
whisker_value1=Q1-(1.5*IQR)
whisker_value2=Q3+(1.5*IQR)

print("whisker 1 and whisker 2 :",whisker_value1,whisker_value2)

X["residual sugar"].loc[X["residual sugar"]<0.84]=Q1

X["residual sugar"].loc[X["residual sugar"]>3.65]=Q3


# In[28]:


# After handling Outliers
plt.boxplot(X["residual sugar"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Residual Sugar')


# In[29]:


# Before handling Outliers
plt.boxplot(X["total sulfur dioxide"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Total sulfur dioxide')


# In[30]:


X["total sulfur dioxide"].quantile(0.25),X["total sulfur dioxide"].quantile(0.75)
IQR=X["total sulfur dioxide"].quantile(0.75)-X["total sulfur dioxide"].quantile(0.25)
print("IQR :",IQR)

# whisker values
Q1=X["total sulfur dioxide"].quantile(0.25)
Q3=X["total sulfur dioxide"].quantile(0.75)
print("Q1 and Q3 : ",Q1,Q3)
whisker_value1=Q1-(1.5*IQR)
whisker_value2=Q3+(1.5*IQR)

print("whisker 1 and whisker 2 :",whisker_value1,whisker_value2)

X["total sulfur dioxide"].loc[X["total sulfur dioxide"]<-39.5]=Q1

X["total sulfur dioxide"].loc[X["total sulfur dioxide"]>124.5]=Q3


# In[31]:


# After handling Outliers
plt.boxplot(X["total sulfur dioxide"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Total sulfur dioxide')


# In[32]:


# Before handling Outliers
plt.boxplot(X["free sulfur dioxide"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Free sulfur dioxide')


# In[33]:


X["free sulfur dioxide"].quantile(0.25),X["free sulfur dioxide"].quantile(0.75)
IQR=X["free sulfur dioxide"].quantile(0.75)-X["free sulfur dioxide"].quantile(0.25)
print("IQR :",IQR)

# whisker values
Q1=X["free sulfur dioxide"].quantile(0.25)
Q3=X["free sulfur dioxide"].quantile(0.75)
print("Q1 and Q3 : ",Q1,Q3)
whisker_value1=Q1-(1.5*IQR)
whisker_value2=Q3+(1.5*IQR)

print("whisker 1 and whisker 2 :",whisker_value1,whisker_value2)

X["free sulfur dioxide"].loc[X["free sulfur dioxide"]<-14]=Q1

X["free sulfur dioxide"].loc[X["free sulfur dioxide"]>42]=Q3


# In[34]:


# After handling Outliers
plt.boxplot(X["free sulfur dioxide"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Free sulfur dioxide')


# In[35]:


# Before handling Outliers
plt.boxplot(X["volatile acidity"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('volatile acidity')


# In[36]:


X["volatile acidity"].quantile(0.25),X["volatile acidity"].quantile(0.75)
IQR=X["volatile acidity"].quantile(0.75)-X["volatile acidity"].quantile(0.25)
print("IQR :",IQR)

# whisker values
Q1=X["volatile acidity"].quantile(0.25)
Q3=X["volatile acidity"].quantile(0.75)
print("Q1 and Q3 : ",Q1,Q3)
whisker_value1=Q1-(1.5*IQR)
whisker_value2=Q3+(1.5*IQR)

print("whisker 1 and whisker 2 :",whisker_value1,whisker_value2)


# In[37]:


X["volatile acidity"].loc[X["volatile acidity"]<0.015]=Q1

X["volatile acidity"].loc[X["volatile acidity"]>1.015]=Q3


# In[38]:


# After handling Outliers
plt.boxplot(X["volatile acidity"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('volatile acidity')


# In[39]:


# Before handling Outliers
plt.boxplot(X["chlorides"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('chlorides')


# In[40]:


X["chlorides"].quantile(0.25),X["chlorides"].quantile(0.75)
IQR=X["chlorides"].quantile(0.75)-X["chlorides"].quantile(0.25)
print("IQR :",IQR)

# whisker values
Q1=X["chlorides"].quantile(0.25)
Q3=X["chlorides"].quantile(0.75)
print("Q1 and Q3 : ",Q1,Q3)
whisker_value1=Q1-(1.5*IQR)
whisker_value2=Q3+(1.5*IQR)

print("whisker 1 and whisker 2 :",whisker_value1,whisker_value2)


# In[41]:


X["chlorides"].loc[X["chlorides"]<0.038]=Q1

X["chlorides"].loc[X["chlorides"]>0.122]=Q3


# In[42]:


# After handling Outliers
plt.boxplot(X["chlorides"], vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('chlorides')


# In[ ]:





# In[43]:



X.shape,y.shape


# In[44]:


X.head()


# # Balanced target variable y
# (unique,count)=np.unique(y,return_counts=True)
# frequencies=np.asarray((unique,count)).T
# print(frequencies)

# # Split data in Training and test data

# In[45]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[46]:


X_train.shape,y_train.shape


# In[47]:


X_train.head()


# # SMOTE - Class Imbalance

# from collections import Counter
# # check version number
# import imblearn
# from imblearn.over_sampling import SMOTE
# print("The number of classes before fit {}".format(Counter(y_train)))
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)
# print("The number of classes after fit {}".format(Counter(y_train)))

# # lets plot SMOTE
# pd.value_counts(y_train).plot.bar()
# plt.title("SMOTE- Class Balance")
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# pd.value_counts(y_train)
# #oversample.plt.bar(X_train, Counter(y_train))

# # Feature Selection

# In[48]:


# ANOVA feature selection for numeric input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
# generate dataset
#X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
# define feature selection
fs = SelectKBest(score_func=mutual_info_classif, k=11) # f_classif


# In[49]:


# apply feature selection
print(fs)
X_train = fs.fit_transform(X_train, y_train)
print(X_train.shape)


# In[ ]:





# In[50]:


# ANOVA correlation coefficient (linear) -Selected top 5 important features
X_train
# type	volatile acidity	chlorides	density	alcohol


# # Feature Scaling

# In[51]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[52]:


scaler=StandardScaler()
#scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_train


# # Model Creation using Naive Bayes

# In[53]:


from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes importMultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, classification_report


# In[54]:


gnb = GaussianNB()
#gnb =MultinomialNB()
gnb.fit(X_train, y_train)


# In[55]:


print("Training Accuracy :",gnb.score(X_train, y_train))


# In[56]:


# Feature Selection 
X_test = fs.transform(X_test)
print(X_test.shape)


# In[57]:


# Feature Scaling on X_test
X_test=scaler.transform(X_test)
X_test


# In[58]:


y_pred_gnb = gnb.predict(X_test)
#y_prob_pred_gnb = gnb.predict_proba(X_test)
# how did our model perform?
#count_misclassified = (y_test != y_pred_gnb).sum()

print("GaussianNB")
#print("=" * 30)
#print('Misclassified samples: {}'.format(count_misclassified))
accuracy = accuracy_score(y_test, y_pred_gnb)
print('Accuracy: {:.2f}'.format(accuracy))


# In[59]:


#Recall, Precision, and F1 score for GaussianNB

print("Recall score : ", recall_score(y_test, y_pred_gnb , average='micro'))
print("Precision score : ",precision_score(y_test, y_pred_gnb , average='micro'))
print("F1 score : ",f1_score(y_test, y_pred_gnb , average='micro'))


# In[60]:


#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, y_train)
print('Random Forest Classifier Training Accuracy:', forest.score(X_train, y_train))
y_pred_rand=forest.predict(X_test)
print('Random Forest Classifier Testing Accuracy:', forest.score(X_test, y_test))
accuracy = accuracy_score(y_test, y_pred_rand)
print(accuracy)


# In[61]:


#classification_report for GaussianNB

print(classification_report(y_test, y_pred_gnb))


# In[62]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_gnb)

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix with labels using Naive Bayes\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Low','High'])
ax.yaxis.set_ticklabels(['Low','High'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[63]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rand)

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix with labels using Random Forest\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Low','High'])
ax.yaxis.set_ticklabels(['Low','High'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[64]:


#classification_report for Random Forest

print(classification_report(y_test, y_pred_rand))


# In[65]:


from collections import Counter
print(Counter(y_test))
print(Counter(y_pred_gnb))
print(Counter(y_pred_rand))


# In[ ]:





# In[ ]:





# In[ ]:




