import pandas as pd

# GOAL:
# need to read the data file and create TFIDF vectors for queries and abstracts

# format of data
# ARTIST, SONG, LYRICS, GENRE

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
    

# we need to separate queries and abstracts
# take 300 (arbitrary number, is there a better number?) songs in development to be query songs
# the rest of the songs, combine similar genre songs to make a complete abstract
    # an abstract can consist of 50 songs with the same genre (this is an parbitrary number? is there a number that it should acc be?)
# can also experiment with just song similarity by just making a single song an abstract

# then we'll have abstracts (consisting of combined lyrics) that are labeled with a genre
# and query songs
    # make an answer key for the query songs
        # a file that on every line consists of "song name, genre"
# compare query songs to every abstract and get cosine similarity scores, sort by highest
    # make an output file that consists of "song name, genre, score"
    # genre maybe comes from a dictionary that maps an abstract to its genre
# go through this output file and produce a final output file that is similar to answer key in format "song name, genre"
    # pick the highest scores in the previous file
# score the model by comparing to the answer key


if __name__ == "__main__":
    main()