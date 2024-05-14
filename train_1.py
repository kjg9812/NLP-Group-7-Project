# This implementation will use word2vec word embeddings instead of TFIDF for the vector similarity task

from gensim.models import Word2Vec
import pandas as pd
import kjg812_create_vector as helper
# from sklearn.metrics.pairwise import cosine_similarity

def main():
    # devote 20% of the data for testing
    # the other 80% can be for development/training
    df = pd.read_csv("df_lyrics.csv")

    # need to shuffle the data
    df = df.sample(frac=1, random_state=42)
    # Reset the index apipfter shuffling
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

    # 13 percent of total are the query songs
    query_songs = test_df.head(300)

    # # Remove the first 300 rows from the original DataFrame
    test_df = test_df.iloc[300:]

    # Reset the index of the dataframes
    query_songs.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    # debug some counts for each genre
    print(train_df['Genre'].value_counts())

    # TRAIN MODEL
    ### for each genre, train model
    grouped = train_df.groupby('Genre')
    genre_models = {}

    # Iterate over the groups
    for genre, group_df in grouped:
        allLyrics = group_df['Lyrics'].str.cat(sep=' ')
        cleanedlyrics = helper.preprocessText(allLyrics)
        model = Word2Vec(cleanedlyrics, vector_size=100, window=5, min_count=1, workers=4)
        # Save the trained model to disk
        model.save(f"{genre}_word2vec.model")
        genre_models[genre] = model


    # TEST MODEL
    # Calculate genre embeddings
    genre_embeddings = {}

    grouped = test_df.groupby('Genre')
    # Iterate over the groups
    for genre, group_df in grouped:
        allLyrics = group_df['Lyrics'].str.cat(sep=' ')
        cleanedlyrics = helper.preprocessText(allLyrics)
        genre_embeddings[genre] = model.infer_vector(cleanedlyrics, genre_models[genre])
    
    ### iterate through query songs and do cosine similarity
    genre_similarity_scores = {}
    for index,row in query_songs.iterrows():
        id = row[1]
        lyrics = row[2]

        # Preprocess lyrics of the test song
        preprocessed_test_song_lyrics = helper.preprocess_text(lyrics)

        # Calculate document embedding for the test song for each genre
        test_song_embeddings = {}
        for genre, model in genre_models.items():
            test_song_embeddings[id] = model.infer_vector(preprocessed_test_song_lyrics, model)

        # Calculate cosine similarity between the test song embedding and genre embeddings
        for genre, embedding in genre_embeddings.items():
            print(test_song_embeddings[id], embedding)
            similarity_score = helper.cosineSim(test_song_embeddings[id], embedding)[0][0]
            genre_similarity_scores[test_song_embeddings[0]] = similarity_score


    

if __name__ == "__main__":
    main()