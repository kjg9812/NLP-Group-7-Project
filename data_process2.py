# This implementation will use word2vec word embeddings instead of TFIDF for the vector similarity task

from gensim.models import Word2Vec
import pandas as pd

def main():
    # devote 20% of the data for testing
    # the other 80% can be for development/training
    df = pd.read_csv("df_lyrics.csv")

    # need to shuffle the data
    df = df.sample(frac=1, random_state=42)
    # Reset the index after shuffling
    df.reset_index(drop=True, inplace=True)

    # then devote 20% to testing
    # Calculate the number of rows for the test set (20% of the original DataFrame)
    test_size = int(0.2 * len(df))

    # Take a random sample of 20% of the rows for the test set
    test_df = df.sample(n=test_size, random_state=42)

    # Create the training set (80% of the original DataFrame)
    train_df = df.drop(test_df.index)

    # Reset the index of the training and test sets
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print(train_df.shape[0])
    # 13 percent of total are the query songs

    # we need to separate queries and abstracts
    # take 300 (arbitrary number, is there a better number?) songs in training to be query songs
    # Take the first 300 rows from the original DataFrame
    # query_songs = train_df.head(9600)

    # # Remove the first 300 rows from the original DataFrame
    # train_df = train_df.iloc[9600:]

    # Reset the index of the dataframes
    query_songs.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    # the rest of the songs, combine similar genre songs to make a complete abstract
        # an abstract can consist of 50 songs with the same genre (this is an parbitrary number? is there a number that it should acc be?)
        # is there any detriment to using all the songs in the genre to represent one abstract?
    # can also experiment with just song similarity by just making a single song an abstract

    # debug some counts for each genre
    print(train_df['Genre'].value_counts())


if __name__ == "__main__":
    main()