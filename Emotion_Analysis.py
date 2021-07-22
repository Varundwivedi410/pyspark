#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import emoji 
import regex
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
import numpy as np
import re
import inflect 
import string
from textblob import TextBlob
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading data form the .csv file
df = pd.read_csv('ISEAR.csv')
  
# shape of dataset 
#print("Shape:", df.shape)

#df = df.drop('id', axis=1)
#df = df.drop('No', axis=1)

# column names 
#print("\nFeatures:", df.columns) 


# In[3]:


#for graph rep.
#plot_size = plt.rcParams["figure.figsize"] 
#print(plot_size[0]) 
#print(plot_size[1])

#plot_size[0] = 8
#plot_size[1] = 6
#plt.rcParams["figure.figsize"] = plot_size


# In[4]:


#Graph showing the consistency in the dataset
#df.Emotion.value_counts().plot(kind='pie', autopct='%1.0f%%')


# In[5]:


#Preprocessing Part.

# Making all letters lowercase
df['Comment']=df['Comment'].str.lower()


# In[6]:


# convert number into words 
p = inflect.engine() 

def convert_number(text): 
    # split string into list of words 
    temp_str = text.split() 
    # initialise empty list 
    new_string = [] 
  
    for word in temp_str: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = p.number_to_words(word) 
            new_string.append(temp) 
  
        # append the word as it is 
        else: 
            new_string.append(word) 
  
    # join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    return temp_str

df['Comment']=df['Comment'].apply(lambda x:convert_number(x))


# In[7]:


# Remove numbers 
def remove_numbers(text): 
    result = re.sub(r'\d+', '', text) 
    return result 
df['Comment']=df['Comment'].apply(lambda x:remove_numbers(x))

# remove punctuation 
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 
df['Comment']=df['Comment'].apply(lambda x:remove_punctuation(x))

# remove whitespace from text 
def remove_whitespace(text): 
    return  " ".join(text.split()) 
df['Comment']=df['Comment'].apply(lambda x:remove_whitespace(x))


# In[8]:


#Removing the common words from the dataset.
freq = pd.Series(' '.join(df['Comment']).split()).value_counts()[:10]
#freq


# In[9]:


freq = list(freq.index)
df['Comment'] = df['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#df['Comment'].head()


# In[10]:


#Removing the un-common words from the dataset.
freq = pd.Series(' '.join(df['Comment']).split()).value_counts()[-10:]
#freq


# In[11]:


freq = list(freq.index)
df['Comment'] = df['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#df['Comment'].head()


# In[12]:


#Correcting the words.
df['Comment'][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[13]:


#Collecting the stopwords.
stop = set(stopwords.words('english'))

#Convert a collection of raw documents to a matrix of TF-IDF features.
vectorizer= TfidfVectorizer(use_idf=True,lowercase=True,strip_accents="ascii",stop_words=stop)


# In[14]:


#Encoding output labels as follow:
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(df.Emotion.values)


# In[15]:


#Inversing the original value of the labels.
y=lbl_enc.inverse_transform(y)
#y


# In[16]:


#Training vectors.
y=df.Emotion


# In[17]:


#Target values.
x=vectorizer.fit_transform(df.Comment)


# In[18]:


#shape of each training and target value.
#print(y.shape)
#print(x.shape)


# In[19]:


# Splitting into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.1)


# In[20]:


#Training the model using Multinomial naive bayes algorithm.
clf=naive_bayes.MultinomialNB()
clf.fit(X_train,y_train)


# In[21]:


#Predicting the emotion of the text using our already trained model
emoji_present=0
print("Enter the line for prediction.")
pred_input=input()
#"I am very happy today! The atmosphere looks cheerful"
m=list([pred_input])

#pre-processing part of the input taken.
m_v=vectorizer.transform(m)

#putting the input into model already trained.
new_value=clf.predict(m_v)

print("The Emotion of the line is:")
print(clf.predict(m_v))

print("\n\n\n")
# In[22]:


# making predictions on the testing set
y_pred = clf.predict(X_test) 


# In[23]:


# comparing actual response values (y_test) with predicted response values (y_pred) 
#print("Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
#results = confusion_matrix(y_test, y_pred) 
#  
#print('Confusion Matrix :')
#print(results) 
#print('Accuracy Score :',accuracy_score(y_test, y_pred)) 
#print('Report : ')
#print(classification_report(y_test, y_pred)) 


# In[24]:


#for emoji part.

def split_count(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)
    return emoji_list

#line = ["😀 🤔  Hello how are you?"]

counter = split_count(m[0])
if len(counter)>0:
    emoji_present=1
#print(' '.join(emoji for emoji in counter))
new_value1=[]
for x in counter:
    if(x=="😀" or x=="😁" or x=="😂" or x=="🤣" or x=="😃" or x=="😄" or x=="😅" or x=="😆"):
        new_value1="joy"
    if(x=="😬" or x=="😠" or x=="😐" or x=="😑" or x=="😡" or x=="😣" or x=="😤" or x=="😾" or x=="🤬"):
        new_value1="anger"
    if(x=="💩"):
        new_value1="disgust"
    if(x=="😮" or x=="🥶" or x=="🥵" or x=="😱" or x=="😰" or x=="😨" or x=="😧" or x=="😦" or x=="😲"):
        new_value1="fear"
    if(x=="😕" or x=="😔" or x=="😢" or x=="😭" or x=="😫" or x=="😨" or x=="🥺" or x=="🤒" or x=="😪"):
        new_value1="sad"
    if(x=="🤗" or x=="🤩" or x=="🤪" or x=="😲"):
        new_value1="surprise"   


# In[25]:


#print the emoji is present.

if (emoji_present==1):
    print("Emoji is present in the statement.\nHence the Emotion of the statement can be as below:\n")
    if(new_value[0]==new_value1):
        print(new_value)
    else:
        print("Emotion of the emoji is different from the emotion of the statement.\n")
        print("Emotion of statement:",new_value[0],"\nEmotion of the emoji:",new_value1)
        print("\nTherefore this can be a sarcastic comment")


# In[26]:


#for training the model with another algorithm.
# Model 2: Linear SVM
#lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
#lsvm.fit(X_train, y_train)
#y_pred = lsvm.predict(X_test)
#print('lsvm accuracy %s' % accuracy_score(y_test, y_pred))


# In[27]:


#For combining the models trained with same dataset.
#seed=7
#kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
# create the sub models
#estimators = []
#estimators.append(('naive', clf))
#estimators.append(('svm', lsvm))
# create the ensemble model
#ensemble = VotingClassifier(estimators)
#results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
#print(results.mean())

