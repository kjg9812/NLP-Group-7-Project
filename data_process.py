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

    print(train_df.shape[0])
    # 13 percent of total are the query songs

    # we need to separate queries and abstracts
    # take 300 (arbitrary number, is there a better number?) songs in training to be query songs
    # Take the first 300 rows from the original DataFrame
    query_songs = train_df.head(300)

    # Remove the first 300 rows from the original DataFrame
    train_df = train_df.iloc[300:]

    # Reset the index of the dataframes
    query_songs.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    # the rest of the songs, combine similar genre songs to make a complete abstract
        # an abstract can consist of 50 songs with the same genre (this is an parbitrary number? is there a number that it should acc be?)
        # is there any detriment to using all the songs in the genre to represent one abstract?
    # can also experiment with just song similarity by just making a single song an abstract

    # debug some counts for each genre
    print(train_df['Genre'].value_counts())

    # then we'll have abstracts (consisting of combined lyrics) that are labeled with a genre
    # and query songs
        # make an answer key for the query songs
            # a file that on every line consists of "song name, genre"
    
    # vectors for queries
    queryVectors = 
    


    # FOR KEVIN H
    # KEVIN!!! INSTEAD OF LIST OF LISTS IM GONNA HAVE A DICTIONARY
        # IT WILL BE KEY = SONG NAME, VALUE = VECTOR
        # so if you index queryVectors["Take Me Home Country Roads"] you will get -> a dicitionary vector
        # each dictionary vector will have KEY = WORD, VALUE = TFIDF SCORE
            # so if you index vector["the"] you will get -> a scalar like 0.5
    # assume you have a list of lists (each sub list is a vector) for both queries and abstracts
    # compare query songs to every abstract and get cosine similarity scores, sort by highest
        # make an output file that consists of "song name, genre, score"
        # genre maybe comes from a dictionary that maps an abstract to its genre
    # go through this output file and produce a final output file that is similar to answer key in format "song name, genre"
        # pick the highest scores in the previous file
    # score the model by comparing to the answer key

    # format of answer key and final output file
    # each line:
    # song1 pop
    # song2 rock
    #....


if __name__ == "__main__":
    main()