import keras
import os
import tensorflow as tf
from transformers import BertweetTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd

### Check if GPU is available
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### Load data id
data_identification = pd.read_csv("data/data_identification.csv")

### Load tweets dataset
tweets_DM = pd.read_json("data/tweets_DM.json", lines=True, dtype=False)
tweets = tweets_DM._source
tweets = pd.json_normalize(tweets)
tweets = tweets.rename(columns={"tweet.hashtags":"hashtags", "tweet.tweet_id":"tweet_id", "tweet.text":"text"})
tweets = tweets.drop(["hashtags"], axis=1)
tweets = pd.merge(tweets, data_identification, on="tweet_id", how="left")

### Clear special symbol
tweets['text'] = tweets['text'].str.replace(' ', '')
tweets['text'] = tweets['text'].str.replace('#', '')

### Get and save tweets test dataset
df_test = tweets[tweets["identification"] == "test"]
df_test = df_test.drop(columns=['identification'])
df_test.to_pickle("data/df_test.pkl")
df_test = df_test[:100]

# The length of each tweets to inference, rest will be truncated.
seq_len = 128
num_samples = len(df_test)
seq_len, num_samples

# Initialize tokenizer
tokenizer = BertweetTokenizer.from_pretrained('vinai/bertweet-base', nomralization=True)

# Tokenize the tweets and returning Numpy tensors
tokens = tokenizer(df_test['text'].tolist(), max_length=seq_len, truncation=True,
                   padding='max_length', add_special_tokens=True,
                   return_tensors='tf')

# The required input for a bert model, input ids, token type id attention masks
tokens.keys()

# Need to transform into tensorflow tensor type
Xids = tokens['input_ids']
Xmask = tokens['attention_mask']
Xtoktype = tokens['token_type_ids']
Xids = tf.cast(Xids, 'float64')
Xmask = tf.cast(Xmask, 'float64')
Xtoktype = tf.cast(Xtoktype, 'float64')
del tokens

dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask))

def map_func(input_ids, masks):
    # We convert our three-item tuple into a two-item tuple where the input item is a dictionary
    return {'input_ids': input_ids, 'attention_mask': masks}

batch_size = 16
# then we use the dataset map method to apply this transformation
dataset = dataset.map(map_func)
dataset = dataset.batch(batch_size, drop_remainder=False)

# Save token test dataset
tf.data.experimental.save(dataset, 'data/test_ds')
test_ds = tf.data.experimental.load('data/test_ds')

# Load trained BERT model
model = tf.keras.models.load_model('DM_model_Bert')
result = model.predict(test_ds, batch_size=None)

# Result list
result_cat = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

# Change result vector into result category
def my_func(a):
    max_index = np.argmax(a)
    return result_cat[max_index]
result_str = np.apply_along_axis(my_func, 1, result)

# Delete unneeded column
df_test["emotion"] = result_str
df_test = df_test.drop(["text"], axis=1)
df_test = df_test.rename(columns={"tweet_id":"id"})
df_test = df_test.drop_duplicates(subset=["id"], keep=False)

# Save result.csv to local storage
print(df_test)
print(df_test.shape)
df_test.to_csv('result.csv', index=False)