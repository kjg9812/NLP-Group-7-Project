import re
from stop_list import closed_class_stop_words
import nltk.tokenize as nltk

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