from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')
import gensim.downloader as api
from collections import Counter
import pandas as pd

def preprocess_lyrics(lyrics):
  lyrics = lyrics.lower()
  # Remove punctuation
  punct = set(string.punctuation)
  lyrics = ''.join(c for c in lyrics if c not in punct)
  # Remove stop words (optional)
  lyrics = [word for word in lyrics.split() if word not in stop]
  words = lyrics.split()
  return words

def create_song_features(lyrics, word2vec_model):
  """
  Creates average word vector representation for a song using a pre-trained word2vec model.

  Args:
      lyrics: String containing the song lyrics.
      word2vec_model: Gensim word2vec model.

  Returns:
      Average word vector representing the song, or None if no word vectors found.
  """
  # Preprocess lyrics
  words = preprocess_lyrics(lyrics)
  # Create word vectors
  word_vectors = [word2vec_model.wv.get_vector(word, None) for word in words]
  # Check if any words were not found in the model
  word_counts = Counter(word for word in word_vectors if word is None)
  if word_counts:
    print(f"WARNING: Words not found in model: {word_counts}")
  # Calculate average word vector (representing the song)
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
  """
  Trains a multi-class classification model for genre prediction using average word vectors.

  Args:
      lyrics_data: List of strings, each containing song lyrics.
      genre_labels: List of integers, corresponding genre labels for each song.
      word2vec_model: Gensim word2vec model.

  Returns:
      Trained genre classification model.
  """
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
  accuracy = accuracy_score(y_test, model.predict(X_test))
  print(f"Model Accuracy: {accuracy:.4f}")
  return model



# Example usage
# Assuming you have loaded your song lyrics data (lyrics_data) which is a list of songs and genre labels (genre_labels) which is a list of labels (1-4)
# and a pre-trained word2vec model (word2vec_model)
df = pd.read_csv("df_lyrics.csv")
# need to shuffle the data
df = df.sample(frac=1, random_state=42)
# Reset the index after shuffling
df.reset_index(drop=True, inplace=True)

# assign numerical values to each genre for classification
df['genre_code'] = df['Genre'].astype('category').cat.codes + 1
coding = {1:"Country", 2: "Pop", 3: "Rap", 4: "Rock"}

song_names = df["Song"]
lyrics_data = df["Lyrics"]
genre_labels = df["genre_code"]

# pretrained model
word2vec_model = api.load("GoogleNews-vectors-negative300.bin")

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