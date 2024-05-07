from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string
from nltk.corpus import stopwords
import gensim.downloader as api
from collections import Counter
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import regex as re

stop = stopwords.words('english')

def preprocess_lyrics(lyrics):
  lyrics = lyrics.lower()
  # Remove punctuation

  # lyrics = re.sub(r"\[(.*?)\]|\((.*?)\)", "", lyrics)

  punct = set(string.punctuation)
  lyrics = ''.join(c for c in lyrics if c not in punct)
  # Remove stop words (optional)
  lyrics = [word for word in lyrics.split() if word not in stop]
  # print(lyrics)
  return lyrics

def create_song_features(lyrics, word2vec_model):
  # Preprocess lyrics
  words = preprocess_lyrics(lyrics)
  # Create word vectors
  word_vectors = [word2vec_model.get_vector(word) for word in words if word in word2vec_model]
  song_vector = average_word_vectors(word_vectors)
  return song_vector

def average_word_vectors(word_vectors):
  if not word_vectors:
    return None
  # Filter out None vectors (words not found in the model)
  word_vectors = [v for v in word_vectors if v is not None]
  # Calculate average vector
  average_vector = sum(word_vectors) / len(word_vectors)
  return average_vector


def train_genre_classifier(lyrics_data, genre_labels, word2vec_model):
  # Create average word vector features for each song
  song_features = [create_song_features(lyrics, word2vec_model) for lyrics in lyrics_data]
  # Filter out songs with no word vector representation (words not found in model)
  song_features = [f for f in song_features if f is not None]
  genre_labels = [l for l, f in zip(genre_labels, song_features) if f is not None]
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(song_features, genre_labels, test_size=0.2, random_state=42)
  # Train a Logistic Regression model (you can choose other models as well)
  model = LogisticRegression(multi_class='ovr', solver='lbfgs')
  model.fit(X_train, y_train)
  # Evaluate model accuracy on test set
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  print(f"Model Accuracy: {accuracy:.4f}")


  cm = metrics.confusion_matrix(y_test, predictions)
  print(cm)
  plt.figure(figsize=(9,9))
  sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
  plt.ylabel('Actual label');
  plt.xlabel('Predicted label');
  all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
  plt.title(all_sample_title, size = 15)
  plt.savefig("log_regression.png")
  return model



# Example usage
# Assuming you have loaded your song lyrics data (lyrics_data) which is a list of songs and genre labels (genre_labels) which is a list of labels (1-4)
# and a pre-trained word2vec model (word2vec_model)

# smaller dataset
# df = pd.read_csv("df_lyrics.csv")

# larger dataset
df = pd.read_csv("song_lyrics.csv",nrows=4000000)
df = df[df["language"] == "en"]
df = df.rename(columns={"tag": "Genre"})
df = df.rename(columns={"lyrics": "Lyrics"})
df = df[df["Genre"] != "misc"]
df = df[df["Genre"] != "rb"]

# Group by 'genre'
grouped = df.groupby('Genre')
# Create an empty DataFrame to store the balanced dataset
balanced_df = pd.DataFrame()
# Iterate over each group
for genre, group_df in grouped:
    # Sample 10,000 entries from each group
    sampled_df = group_df.sample(n=min(50000, len(group_df)))
    # Append sampled data to balanced_df
    balanced_df = pd.concat([balanced_df, sampled_df])
# Reset the index of the balanced DataFrame
balanced_df.reset_index(drop=True, inplace=True)
print(balanced_df["Genre"].value_counts())

# # need to shuffle the data
balanced_df = balanced_df.sample(frac=1, random_state=42)
# Reset the index after shuffling
balanced_df.reset_index(drop=True, inplace=True)


# assign numerical values to each genre for classification
balanced_df['genre_code'] = balanced_df['Genre'].astype('category').cat.codes + 1
print(balanced_df['genre_code'].head(5))

coding = {1:"Country", 2: "Pop", 3: "Rap", 4: "Rock"}

# song_names = df["Song"].tolist()
lyrics_data = balanced_df["Lyrics"].tolist()
genre_labels = balanced_df["genre_code"].tolist()
# print(lyrics_data)
# pretrained model
from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

model = train_genre_classifier(lyrics_data, genre_labels, word2vec_model)


### TESTS
# # Use the trained model to predict genre for new songs
# new_lyrics = """ 
# This is a new song, can you predict the genre?
# """
# new_song_features = create_song_features(new_lyrics, word2vec_model)
# predicted_genre = model.predict(new_song_features)[0]

# # Based on your genre label mapping, interpret the predicted genre
# print(f"Predicted Genre for the new song: {predicted_genre}")