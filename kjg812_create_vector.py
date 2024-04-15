import re
from stop_list import closed_class_stop_words
import nltk.tokenize as nltk
import math

# preprocess text function
def preprocessText(words):
    # remove punctuation
    words = re.sub(r'[^\w\s-]', '', words)
    # remove numbers
    words = re.sub(r'\d+', '', words)
    # tokenize
    words = nltk.word_tokenize(words)
    # remove stop words
    words = [word for word in words if word not in closed_class_stop_words]
    # split hyphen words
    final = []
    for word in words:
        if "-" in word:
            final.extend(word.split("-"))
        else:
            final.append(word)
    return final

# function that creates a vector representation
# takes in words (lyrics of a song or grouped lyrics of a genre)
# takes in idf scores
def createVector(words, idfs):
    counts = {}
    sentence = preprocessText(words)
    for word in sentence:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    vector = {}
    for word in counts:
        vector[word] = (counts[word]/len(counts)) * idfs[word]
    return vector

# function that returns a dictionary of idf scores for each word that exists in queries and abstracts
# takes in queries (query songs that we are classifying)
# takes in abstracts (grouped songs that represent genres)
def calculateIDF(queries, abstracts):
    # get all words
    globalwords = set()
    for query in queries:
        globalwords.update(preprocessText(query.split('.W')[-1]))
    for abstract in abstracts:
        globalwords.update(preprocessText(abstract.split('.W')[-1]))

    # count frequencies
    abstractfrequencies = {}
    for abstract in abstracts:
        text = preprocessText(abstract.split('.W')[-1])
        for word in globalwords:
            if word in text:
                if word not in abstractfrequencies:
                    abstractfrequencies[word] = 1
                else:
                    abstractfrequencies[word] += 1

    # then IDF score
    idfs = {}
    length = len(abstracts)
    for word in globalwords:
        if word not in abstractfrequencies:
            idfs[word] = 0
        else:
            idfs[word] = math.log(length/(abstractfrequencies[word] + 1))
    
    return idfs

# cosine similarity function
# takes in two vectors, pumps out their cosine similarity
def cosineSim(vec1,vec2):
    dotproduct = 0
    for word in vec1:
        if word in vec2:
            dotproduct += vec1[word] * vec2[word]
    mag1 = math.sqrt(sum(vec1[word]**2 for word in vec1))
    mag2 = math.sqrt(sum(vec2[word]**2 for word in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dotproduct / (mag1*mag2)