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
    # Take the first 9600 rows from the original DataFrame

    
query_songs = train_df.head(9600)

    # Remove the first 9600 rows from the original DataFrame
train_df = train_df.iloc[9600:]

    # Reset the index of the dataframes
query_songs.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)

#cap df value counts to 12000 (could change to satisfy different size )

print(train_df['Genre'].value_counts())

print(train_df)
genreCounts = train_df["Genre"].value_counts()#df of genre counts and genres
print(genreCounts)
#store upcoming dfs
dfs=[]
ratios = 12000/genreCounts #12000/genre count to get ratio needed to get to 12000 genre count
for genre,ratio in ratios.items():#go thru ratio df by genre
    genreDF=train_df.loc[train_df['Genre']==genre]#grab every row in the current genre and make a df for this genre
    genreDF = genreDF.sample(frac = ratio)#grab a sample of ratio size from genreDF--> gets to 12000 total
    dfs.append(genreDF)#append to list of dfs
balancedDF = pd.concat(dfs,ignore_index=True)#concat all dfs in genre list to new balanced DF
print(balancedDF)
print(balancedDF['Genre'].value_counts()) #confirm that genre counts are all 120000

    # the rest of the songs, combine similar genre songs to make a complete abstract
        # an abstract can consist of 50 songs with the same genre (this is an parbitrary number? is there a number that it should acc be?)
        # is there any detriment to using all the songs in the genre to represent one abstract?
    # can also experiment with just song similarity by just making a single song an abstract

    # debug some counts for each genre
    #set train_df to balancedDF so train_df has genre counts of equal size
train_df=balancedDF
print(train_df['Genre'].value_counts())
    
testdict = {"Country": 0, "Rap": 0, "Rock": 0, "Pop": 0}
totaldict = {"Country": 0, "Rap": 0, "Rock": 0, "Pop": 0}
accuracydict ={"Country": 0, "Rap": 0, "Rock": 0, "Pop": 0}
with open('final_output.txt','r')as f:
    counter = 0
    correct = 0
    for line in f:
        line_list = line.split()#now we can get genre by indexing
        if line_list[-1] == "Country":
            totaldict['Country']+=1
        elif line_list[-1] == "Rap":
            totaldict['Rap']+=1
        elif line_list[-1] == "Pop":
            totaldict['Pop']+=1
        elif line_list[-1] == "Rock":
            totaldict['Rock']+=1
        
        if line_list[-1]==query_songs.iloc[counter]['Genre']: 
            testdict[query_songs.iloc[counter]['Genre']] += 1
            correct +=1
        counter+=1
accuracy = correct/(counter+1)
print(counter+1)
#print(query_songs.head(5))
print(testdict)
print(totaldict)
for key in accuracydict:
    accuracydict[key] = testdict[key]/totaldict[key]
print(accuracydict)
print("ACCURACY= ",accuracy)
