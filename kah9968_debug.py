import pandas as pd
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

with open('final_output.txt','r')as f:
    counter = 0
    correct = 0
    for line in f:
        line_list = line.split()#now we can get genre by indexing
        if line_list[-1]==query_songs.iloc[counter]['Genre']: 
            correct +=1
        counter+=1
accuracy = correct/(counter+1)
print(query_songs.head(5))
print("ACCURACY= ",accuracy)
