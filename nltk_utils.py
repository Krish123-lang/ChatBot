# Import necessary libraries
import nltk
import numpy as np

# nltk.download('punkt')

# Import the PorterStemmer from NLTK
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Tokenize function: Splits a sentence into a list of words (tokens)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stemming function: Reduces words to their root/base form
def stem(word):
    return stemmer.stem(word.lower())

# Bag of Words function: Converts a tokenized sentence into a numerical representation
# where each element of the array corresponds to a word in the vocabulary.
def bag_of_words(tokenized_sentence, all_words):
    # Stem all the words in the tokenized sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Initialize a numpy array filled with zeros
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Iterate through the words in the vocabulary
    for idx, w in enumerate(all_words):
        # If the word is in the tokenized sentence, set the corresponding element to 1.0
        if w in tokenized_sentence:
            bag[idx] = 1.0

    # Return the bag of words representation
    return bag


"""_summary_
1. It imports the necessary libraries, including nltk for natural language processing and numpy for numerical operations.

1. The PorterStemmer from the NLTK library is imported. Stemming is a process of reducing words to their base or root form. For example, "jumping" and "jumps" both get reduced to "jump."

3. The stemmer is initialized using the PorterStemmer().

4. The tokenize function takes a sentence as input and uses NLTK's word_tokenize function to split it into a list of words (tokens). Tokenization is the process of breaking text into individual words or tokens.

5. The stem function takes a word as input, converts it to lowercase, and then applies stemming using the stemmer. It returns the stemmed word.

6. The bag_of_words function takes a tokenized sentence and a list of all words in the vocabulary as input. It first stems all the words in the tokenized sentence to ensure consistency.

7. It initializes a numpy array called bag filled with zeros, where the length of the array is equal to the number of words in the vocabulary (len(all_words)).

8. It iterates through the words in the vocabulary (all_words) and checks if each word is present in the stemmed tokenized sentence. If a word is present, the corresponding element in the bag array is set to 1.0, indicating its presence in the sentence.

9. Finally, the function returns the bag of words representation, where each element in the array corresponds to a word in the vocabulary, and its value indicates whether that word is present (1.0) or not (0.0) in the tokenized sentence.
"""