#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[2]:


age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)


# In[3]:


df.drop("Cabin", axis=1, inplace=True)


# In[4]:


df['Embarked'].fillna("S", inplace=True)


# In[5]:


df.loc[df['Age'] > 70, 'Age'] = 70


# In[6]:


df.loc[df['Fare'] > 400, 'Fare'] = df['Fare'].median()


# In[7]:


df.drop("PassengerId", axis=1, inplace=True)


# In[10]:


def title_name(name):
    if "," in name:
        return name.split(",")[1].split(".")[0].strip()


titles = set([title_name(x) for x in df['Name']])


def shorted_titles(x):
    titles = x["titles"]
    if titles in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif titles in ['Jonkheer', 'Don', 'the Countess', 'Lady', 'Sir']:
        return 'Royalty'
    elif titles == 'Mme':
        return 'Mrs'
    elif titles in ['Mlle', 'Miss', 'Ms']:
        return 'Miss'
    else:
        return titles


df['titles'] = df['Name'].map(lambda x: title_name(x))
df['titles'] = df.apply(shorted_titles, axis=1)


# In[11]:


df.drop("Name", axis=1, inplace=True)


# In[12]:


df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
df['titles'].replace(df["titles"].value_counts().keys(), [
                     0, 1, 2, 3, 4, 5, 6, 7], inplace=True)


# In[13]:


corr = df.corr()
corr.Survived.sort_values(ascending=False)


# In[14]:


# In[15]:


x = df.drop(['Survived', 'Ticket'], axis=1)
y = df['Survived']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)


# In[16]:


randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val)*100,  2)
print("Accuracy of Random Forest:", acc_randomforest)

pickle.dump(randomforest, open('titanic_mode.sav', 'wb'))


# In[22]:


# Correct order in the dataframe:
def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import pickle
    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
    randomforest = pickle.load(open('titanic_mode.sav', 'rb'))
    predictions = randomforest.predict(x)
    print(predictions)


# In[23]:
