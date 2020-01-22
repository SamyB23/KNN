#!/usr/bin/env python
# coding: utf-8

# In[3]:


iris_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


# In[4]:


#Import block
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


names = ['sepal_length','sepal_width','petal_length','petal_width','species']
dataset = pd.read_csv(filepath_or_buffer=iris_data_url,names=names)
dataset


# In[8]:


np.unique(dataset.species)


# In[9]:


#Plot a bar graph for each of the three different flower species and their corresponding numbers from the above dataset
#how is your data distributed?
setosa_counter = 0
versicolor_counter = 0
virginica_counter = 0
frequency = []
for i in dataset["species"]:
    if i == "Iris-setosa":
        setosa_counter += 1
    elif i == "Iris-versicolor":
        versicolor_counter += 1
    elif i == "Iris-virginica":
        virginica_counter += 1

frequency.append(setosa_counter)
frequency.append(versicolor_counter)
frequency.append(virginica_counter)

print(frequency)

plt.bar(np.unique(dataset.species),frequency, color='red')
plt.show()


# In[10]:


#Figure out the average and standard deviation for each of the different attributes, print these out
for column in dataset.columns:
    if column == "species":
        print("Cannot find average or standard deviation of strings")
    else:
        print(column)
        print("Mean =" , dataset[column].mean())
        print("Standard deviation = " , dataset[column].std())
        print("\n")


# In[11]:


y = dataset.species
x = dataset.drop('species',axis=1)


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

##Cross validate to find optimal k (number of neighbors)
##Plot number of neighbors against accuracy
##Sweep k from k=1,40 inclusive, use number of folds = 10
##Choose that optimal k, make a confusion matrix using seaborn (the nicer version)
results = []
classifier_list = []
k_list = list(range(1,41,2))

for i in k_list:
    accuracy_list = []
    
    knn = KNeighborsClassifier(n_neighbors = i)

    kf = KFold(n_splits = 10, random_state=None, shuffle=True)
    
    for train, test in kf.split(X_train, y_train):
        x_train_cv, x_test_cv = X_train.iloc[train], X_train.iloc[test]
        y_train_cv, y_test_cv = y_train.iloc[train], y_train.iloc[test]
        # fitting the model
        current_classifier = knn.fit(x_train_cv, y_train_cv)
    
        classifier_list.append(current_classifier)

        # predict the response
        prediction = knn.predict(x_test_cv)

        current_accuracy=current_classifier.score(x_test_cv, y_test_cv)
        accuracy_list.append(current_accuracy)
        
        
        # evaluate accuracy
        #print (accuracy_score(y_test, prediction))

    results.append(np.mean(accuracy_list))
    print(np.mean(accuracy_list))
    
best_value_index = accuracy_list.index(max(accuracy_list))
classifier_list[best_value_index]
plt.plot(k_list,results)
plt.show()


# In[13]:


best_value_index = results.index(max(results))
k_list[best_value_index]

cm = confusion_matrix(y_test_cv, prediction)
print(cm)
sns.heatmap(cm, annot=True, fmt="g", linewidths=.5, square = True, cmap = 'Blues_r');


# In[70]:


##classifcation to n dimensional and n-ary target class values
from collections import Counter

# def predict(X_train,y_train,test_point, k):
#     distances = []
#     knn = []
    
#     for i in range(len(X_train)):
#         distance = np.sum(np.square(test_point - X_train.iloc[i, :]))
#         distance = np.sqrt(distance)
#         distances.append(distance)
        
#     ##SORTING WOULD GO HERE

#     distances = np.argsort(distances)
#     ##stores k neighbors into a list
#     for i in range(k):
#         index = distances[i]
#         ##This creates a list of votes
#         knn.append(y_train.iloc[index])

#     ##gets the most common element from the list of votes for the majority vote
#     frequency = Counter(knn).most_common(1)[0][0]    
#     return frequency

def predict2(X_train,y_train,test_point, k):
    distances = []
    knn = []
    
    kf = KFold(n_splits = 10, random_state=None, shuffle=True)
    
    for train, test in kf.split(X_train, y_train):
        x_train_cv, x_test_cv = X_train.iloc[train], X_train.iloc[test]
        y_train_cv, y_test_cv = y_train.iloc[train], y_train.iloc[test]
        
    for i in range(len(x_train_cv)):
        distance = np.sum(np.square(test_point - x_train_cv.iloc[i, :]))
        distance = np.sqrt(distance)
        distances.append(distance)
        
    ##SORTING WOULD GO HERE
    distances = np.argsort(distances)
    ##stores k neighbors into a list
    for i in range(k):
        index = distances[i]
        ##This creates a list of votes
        knn.append(y_train_cv.iloc[index])

    ##gets the most common element from the list of votes for the majority vote
    frequency = Counter(knn).most_common(1)[0][0]    
    return frequency

# def knn(X_train,y_train,x_test, k, prediction_list):
#     for i in range(len(x_test)):
#         prediction_list.append(predict(X_train, y_train, x_test.iloc[i,:], k))

def knn2(X_train,y_train,x_test, k, prediction_list):
    for i in range(len(x_test)):    
        prediction_list.append(predict2(X_train, y_train, x_test.iloc[i,:], k))


# In[41]:


# for i in range(1,41,2):
#     sample = []
#     knn(X_train, y_train, X_test, i, sample)

# accuracy = accuracy_score(y_test, sample)
# print(accuracy*100, "%")


# In[73]:


for i in range(1,55,2):
    sample2 = []
    knn2(X_train, y_train, X_test, i, sample2)
    
accuracy = accuracy_score(y_test, sample2)
print(accuracy*100, "%")
cm = confusion_matrix(y_test, sample2)
sns.heatmap(cm, annot=True, fmt="g", linewidths=.5, square = True, cmap = 'Blues_r');
best_value_index = sample2.index(max(sample2))
print(best_value_index)


# In[ ]:




