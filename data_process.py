# need to read the data file and create TFIDF vectors for queries and abstracts

# format of data
# ARTIST, SONG, LYRICS, GENRE

# devote 10% of the data for testing
# the other 90% can be for development/training

# need to shuffle the data
# then devote 10% to testing

# we need to separate queries and abstracts
# take 300 (arbitrary number, is there a better number?) songs in development to be query songs
# the rest of the songs, combine similar genre songs to make a complete abstract
    # an abstract can consist of 50 songs with the same genre (this is an arbitrary number? is there a number that it should acc be?)
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




