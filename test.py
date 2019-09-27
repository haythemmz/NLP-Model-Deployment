#%%
import pandas as pd 
import pickle 
#%%
filename="svm_bow.sav"
loaded_model = pickle.load(open(filename, 'rb'))

#%%
def normalizer_all(tweet):
    tweets = " ".join(filter(lambda x: x[0]!= '@' , tweet.split()))
    tweets = re.sub('[^a-zA-Z]', ' ', tweets)
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [word for word in tweets if not word in set(stopwords.words('english'))]
    tweets =' '.join([porter.stem(w) for w in tweets])
    return tweets


clean_message=normalizer_all(message_love)

a=vectorizer.transform([clean_message])

#%%
