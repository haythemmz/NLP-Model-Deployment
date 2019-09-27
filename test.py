#%%
import pandas as pd 
import pickle 
import re 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import nltk 
from nltk.corpus import stopwords

porter = PorterStemmer()
lancaster=LancasterStemmer()
#%%
filename="svm_bow.sav"


#%%
def normalizer_all(tweet):
    tweets = " ".join(filter(lambda x: x[0]!= '@' , tweet.split()))
    tweets = re.sub('[^a-zA-Z]', ' ', tweets)
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [word for word in tweets if not word in set(stopwords.words('english'))]
    tweets =' '.join([porter.stem(w) for w in tweets])
    return tweets

message_love=" I loving man kiss"
clean_message=normalizer_all(message_love)
print(clean_message)


#a=vectorizer.transform([clean_message])

#%%
bow= pickle.load(open("bow.sav", 'rb'))
a=bow.transform([clean_message])
svm_bow= pickle.load(open("svm_bow.sav", 'rb'))
print(svm_bow.predict_proba(a.toarray())[0][0])

#%%
a=bow.transform([clean_message])

#%%
print(a.shape)

#%%
svm_bow= pickle.load(open("knn_bow.sav", 'rb'))

#%%
svm_bow.predict_proba(a.toarray())

#%%
a.toarray().shape

#%%
