import os
import gc
import tensorflow as tf
from transformers import TFAutoModel
from transformers import BertweetTokenizer
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import keras

### Check if GPU is available
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### Load data identification
data_identification = pd.read_csv("data/data_identification.csv")

### Load emotion of data
emotion = pd.read_csv("data/emotion.csv")

### Tweets dataset
tweets_DM = pd.read_json("data/tweets_DM.json", lines=True, dtype=False)
print(tweets_DM.shape)

### Get only tweets_id and text attribute
tweets = tweets_DM._source
tweets = pd.json_normalize(tweets)
tweets = tweets.rename(columns={"tweet.hashtags":"hashtags", "tweet.tweet_id":"tweet_id", "tweet.text":"text"})
tweets = tweets.drop(["hashtags"], axis=1)

### Merge identification of data to tweets
tweets = pd.merge(tweets, data_identification, on="tweet_id", how="left")

### Clear special symbol
tweets['text'] = tweets['text'].str.replace(' ', '')
tweets['text'] = tweets['text'].str.replace('#', '')

### Get train and test dataset
df_train = tweets[tweets["identification"] == "train"]
df_train = df_train.drop(columns=['identification'])
df_train = pd.merge(df_train, emotion, on="tweet_id")

### Check data corruption
df_train.isnull().sum()
df_train.duplicated().sum()
df_train.groupby(['emotion']).count()['text']

### Save data to local storage
df_train.to_pickle("data/df_train.pkl")

### Samples of each categories
### This is almost my limitation, if too large my ram will be out of space
targ = int(70000)

# Form new balanced training dataset
df_anger = df_train[df_train['emotion'] == 'anger']
df_anticipation = df_train[df_train['emotion'] == 'anticipation'].sample(targ)
df_disgust = df_train[df_train['emotion'] == 'disgust'].sample(targ)
df_fear = df_train[df_train['emotion'] == 'fear']
df_joy = df_train[df_train['emotion'] == 'joy'].sample(targ)
df_sadness = df_train[df_train['emotion'] == 'sadness'].sample(targ)
df_surprise = df_train[df_train['emotion'] == 'surprise']
df_trust = df_train[df_train['emotion'] == 'trust'].sample(targ)
df_train_balanced = pd.concat([df_anger, df_anticipation, df_disgust, df_fear, df_joy, df_sadness, df_surprise, df_trust], ignore_index=True)

# Clean up memory to prevent crash
del df_anger
del df_anticipation
del df_disgust
del df_fear
del df_joy
del df_sadness
del df_surprise
del df_trust
del df_train
del targ
del emotion
del tweets
del data_identification
del tweets_DM

# The length of each tweets to inference, rest will be truncated.
seq_len = 128
num_samples = len(df_train_balanced)
print(num_samples)

# Collect garbage to free up some ram
gc.collect()

# Initialize tokenizer
tokenizer = BertweetTokenizer.from_pretrained('vinai/bertweet-base', nomralization=True)

# Tokenize the tweets and returning Numpy tensors
tokens = tokenizer(df_train_balanced['text'].tolist(), max_length=seq_len, truncation=True,
                   padding='max_length', add_special_tokens=True,
                   return_tensors='tf')

# Test word vectors
line = "Donald"
line1 = "#DonaldTrump"
line2 = "@DonaldTrump"
line3 = "DonaldTrump"
tokenizer.encode(line), tokenizer.encode(line1), tokenizer.encode(line2), tokenizer.encode(line3)

# Get required input for a bert model, input ids, token type id attention masks
tokens.keys()

# Need to transform into tensorflow tensor type
Xids = tokens['input_ids']
Xmask = tokens['attention_mask']
Xtoktype = tokens['token_type_ids']
Xids = tf.cast(Xids, 'float64')
Xmask = tf.cast(Xmask, 'float64')
Xtoktype = tf.cast(Xtoktype, 'float64')
del tokens

# Extrace and one hot encode the labels
arr = df_train_balanced['emotion']
le = LabelEncoder()
arr = le.fit_transform(arr)
print(le.classes_)
labels = np.zeros((num_samples, arr.max()+1))
print(labels.shape)
labels[np.arange(num_samples), arr] = 1
print(labels[100])

# Transform ids, masks, labels into a tensorflow dataset type
dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

def map_func(input_ids, masks, labels):
    # we convert our three-item tuple into a two-item tuple where the input item is a dictionary
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

# Then we use the dataset map method to apply this transformation
dataset = dataset.map(map_func)
print(dataset.take(1))

# Set batch size and shuffle data
batch_size = 16
dataset = dataset.shuffle(labels.shape[0]).batch(batch_size, drop_remainder=True)
print(dataset.take(1))

# Do train-val split
split = 0.9
# we need to calculate how many batches must be taken to create 90% training set
size = int((Xids.shape[0] / batch_size) * split)
print(size)
train_ds = dataset.take(size)
val_ds = dataset.skip(size)

# Free up memory
del dataset

# Save arranged data
tf.data.experimental.save(train_ds, 'data/train_ds')
tf.data.experimental.save(val_ds, 'data/val_ds')

# Load pretrained BERTtweets model
bert = TFAutoModel.from_pretrained('vinai/bertweet-base')
bert.summary()

# Two input layers, we ensure layer name variables match to dictionary keys in TF dataset
input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int32')
toktypeid = tf.keras.layers.Input(shape=(seq_len,), name='token_type_ids', dtype='int32')

# Access final activations (alread max-pooled) [1]
embeddings = bert.roberta(input_ids, attention_mask=mask)[1] 

# Convert bert embeddings into 5 output classes
x = tf.keras.layers.Dense(2048, activation='relu')(embeddings)
x1 = tf.keras.layers.Dropout(0.2)(x)
x2 = tf.keras.layers.Dense(64, activation='relu')(x1)
x3 = tf.keras.layers.Dropout(0.2)(x2)
y = tf.keras.layers.Dense(8, activation='softmax', name='outputs')(x)

# Initialize model
model = tf.keras.Model(inputs=[input_ids, toktypeid, mask], outputs=y)
model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

element_spec = ({'input_ids': tf.TensorSpec(shape=(16, seq_len), dtype=tf.float64, name=None),
                 'attention_mask': tf.TensorSpec(shape=(16, seq_len), dtype=tf.float64, name=None)},
                 # 'token_type_ids': tf.TensorSpec(shape=(16, seq_len), dtype=tf.float64, name=None)},
                tf.TensorSpec(shape=(16, 8), dtype=tf.float64, name=None))

# Load the training and validation sets
train_ds = tf.data.experimental.load('data/train_ds', element_spec=element_spec)
val_ds = tf.data.experimental.load('data/val_ds', element_spec=element_spec)
print(train_ds)

# Optimizers to the model
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5, decay=1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    shuffle=True
)

# Save model
model.save('DM_model_3')