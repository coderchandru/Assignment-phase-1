#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install contractions missingno wordcloud')


# In[2]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings(action='ignore')

# Import NLTK and download required resources
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem import LancasterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Import other libraries
import re
import string
import unicodedata
import contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import wordcloud
from wordcloud import STOPWORDS, WordCloud
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    precision_recall_fscore_support
)

# Set options for displaying data
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)


# In[51]:


df = pd.read_csv('Tweets1.csv')
df.head()


# In[52]:


texts = [[word.lower() for word in text.split()] for text in df]


# In[53]:


df.head()


# In[54]:


df.info()


# In[55]:


df.isnull().sum()


# In[56]:


plt.figure(figsize=(10,7))
sns.heatmap(df.isnull(), cmap = "Blues")                       
plt.title("Missing values?", fontsize = 15)
plt.show()


# In[57]:


print("Percentage null or na values in df")
((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)


# In[58]:


df.drop(["tweet_coord", "product_sentiment_gold", "negativereason_gold"], axis=1, inplace=True)


# In[59]:


df.head()


# In[60]:


freq = df.groupby("negativereason").size()


# In[61]:


df.duplicated().sum()


# In[62]:


df.drop_duplicates(inplace = True)


# In[63]:


df.describe().T


# In[64]:


df.nunique()


# In[66]:


plt.figure(figsize=(7,3))
sns.countplot(data=df,x='product', palette=['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#a6cee3'])
plt.show()


# In[69]:


sentiment_counts = df['product_sentiment'].value_counts()
plt.figure(figsize=(6, 4))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of product Sentiments')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[70]:


value_counts = df['negativereason'].value_counts()

# Create a donut-like pie chart using matplotlib and seaborn
plt.figure(figsize=(8, 8))
labels = value_counts.index
values = value_counts.values
colors = sns.color_palette('pastel')[0:len(labels)]  # Use pastel colors for the chart
plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
plt.title('Overall distribution for negative reasons')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is drawn as a circle.
plt.show()


# In[71]:


df = df[['product_sentiment', 'text']].copy()


# In[72]:


df


# In[73]:


X = df["text"]
y = df["product_sentiment"]


# In[74]:


X


# In[75]:


y


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[77]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[78]:


tfidf = TfidfVectorizer(stop_words="english")


# In[79]:


tfidf.fit(X_train)


# In[80]:


print(tfidf.get_feature_names_out())


# In[81]:


print(tfidf.vocabulary_)


# In[82]:


count_vect = CountVectorizer(stop_words="english")
neg_matrix = count_vect.fit_transform(df[df["product_sentiment"]=="negative"]["text"])
freqs = zip(count_vect.get_feature_names_out(), neg_matrix.sum(axis=0).tolist()[0])

print(sorted(freqs, key=lambda x: -x[1])[:100])


# In[83]:


new_df = df[df["product_sentiment"] == "positive"]
words = " ".join(new_df["text"])
cleaned_word = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
wordcloud = WordCloud(stopwords = STOPWORDS,
                     background_color = "black", width = 3000, height = 2500).generate(cleaned_word)
plt.figure(figsize = (12, 12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[33]:


new_df = df[df["airline_sentiment"] == "negative"]
words = " ".join(new_df["text"])
cleaned_word = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
wordcloud = WordCloud(stopwords = STOPWORDS,
                     background_color = "black", width = 3000, height = 2500).generate(cleaned_word)
plt.figure(figsize = (12, 12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[84]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df["product_sentiment"])
df["product_sentiment_encoded"] = le.transform(df["product_sentiment"])
df.head()


# In[85]:


def tweet_to_words(tweet):
    letters_only = re.sub("[^a-zA-Z]", " ", tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return(" ".join( meaningful_words ))


# In[86]:


nltk.download("stopwords")
df["clean_tweet"] = df["text"].apply(lambda x: tweet_to_words(x))


# In[87]:


df.info()


# In[88]:


X = df["clean_tweet"]
y = df["product_sentiment"]


# In[89]:


print(X.shape, y.shape)


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[91]:


vect = CountVectorizer()
vect.fit(X_train)


# In[92]:


X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)


# In[93]:


vect_tunned = CountVectorizer(stop_words = "english", ngram_range = (1, 2), min_df = 0.1, max_df = 0.7, max_features = 100)
vect_tunned


# In[94]:


from sklearn.svm import SVC
model = SVC(kernel = "linear", random_state = 10)
model.fit(X_train_dtm, y_train)
pred = model.predict(X_test_dtm)


# In[95]:


print("Accuracy Score: ", accuracy_score(y_test, pred) * 100)


# In[96]:


print("Confusion Matrix\n\n", confusion_matrix(y_test, pred))


# In[97]:


conf_matrix = pd.DataFrame(data=confusion_matrix(y_test, pred), columns=['Predicted:0', 'Predicted:1', 'Predicted:2'], index=['Actual:0', 'Actual:1', 'Actual:2'])

# Plot the 3x3 confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Paired', cbar=False, linewidths=0.1, annot_kws={'size': 25})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[98]:


print(classification_report(y_test, pred))


# In[ ]:




