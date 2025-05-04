import nltk
nltk.download('words')
from nltk.corpus import words

import numpy as np

def processData():
    """Goal: return a list with dimensions: (batch_size, word length)
    Each word would be converted into a corresponding array of integers"""
    word_list = words.words()

    #shuffle the word_list
    # word_list_shuffled = np.random.permutation(len(word_list))
    
    # #take the first batch_size of words
    # batch = word_list_shuffled[:batch_size]

    # strToInt = []

    # for i in batch:
    #     for j in word_list[i]:
    #         if (ord(j) >= 60 and ord(j) <= 90):
    #             strToInt.append(ord(j)-65)
    #         else:
    #             strToInt.append(ord(j)-97)

    #MORE EFFICIENT BUT UNREADABLE 
    #let a-z be represented by 1-26 in each word, remove caps
    # strToInt = [[(ord(j)-64 if ord(j) >= 60 and ord(j) <= 90 else ord(j)-96) for j in word_list[i]] for i in batch]
    """SHOULD BE USING ENTIRE DATASET TO TRAIN NOT JUST BATCH"""
    strToInt = [[(ord(j)-64 if ord(j) >= 60 and ord(j) <= 90 else ord(j)-96) for j in word_list[i] if j.isalpha()] for i in range(len(word_list))]

    vocab = {chr(i): i-96 for i in range(97, 123)}  # a-z -> 1-26
    vocab['token'] = 0  # Start token
    reverse_vocab = {v: k for k, v in vocab.items()}

    return strToInt, vocab, reverse_vocab

def processConsecutive(data, permutations):
    """Goal: Track relationship between the words, keep track of the previous character
    also create a token forst the first and last letter, otherwise will just have 1 element in a 2 element tuple
    use . for eg -> .a means a is first letter (or more so (., 1) and a. means last letter (or more so (1, .)))"""
    
    #creates a list of list of tuples that capture the relationship between 1 element and its neighbor
    tupleLetters = [[(0,data[j][0])] + [(data[j][i], data[j][i+1]) for i in range(len(data[j])-1)] + [(data[j][len(data[j])-1],0)] for j in range(len(data))]

    #loop thru tupleletters to fill in permutations
    for i in range(len(tupleLetters)):
        for j in range(len(tupleLetters[i])):
            for i1, i2 in tupleLetters[i]:
                permutations[i1][i2]+=1
    
    #calculates probability of each element in row
    for i in range(27):
        row_sum = np.sum(permutations[i])
        if row_sum > 0:
            permutations[i] /= row_sum
                
    return permutations

def generate_word(permutations, vocab, reverse_vocab, max_len, temperature):
    """Generate a new word using the Markov chain."""
    seq = [vocab['token']]
    word = []

    while len(word) < max_len:
        #takes last element of seq list to obtain the list of probabilites for the next character 
        curr=seq[-1]
        probs = permutations[curr].copy()

        #used for skewing probabilities -> all probabilities <= 1, so probability^x <= 1. therefore, exponentiating all probabilities by same exp will change ratios
        #then, normalize
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / np.sum(probs)

        next_char = np.random.choice(len(vocab), p=probs)

        #if ends prematurely, break
        if next_char==vocab['token']:
            break
        
        #append the char to the sequence, to the word reverse the char->int conversion to int->char
        seq.append(next_char)
        word.append(reverse_vocab[next_char])

    #"connects" the word list from list of chars to string and returns
    generated = ''.join(word)
    return generated
    

def main():
    data, vocab, reverse_vocab = processData()

    #creates 2d array has 27 keys, 26 for each letter, 
    #and 1 extra to indicate the token in which the other letter in the tuple is either first or last in str eg. 0,1 or 1,0
    #starts off at 0, for every pattern that is found, add 1 eg -> (18, 1), do permutations[18][1]++
    permutations = np.zeros((27, 27), dtype=float)
    permutations = processConsecutive(data, permutations)

    temperature = 0.5
    max_len = 15
    #generates new words
    for _ in range(5):
        word = generate_word(permutations, vocab, reverse_vocab, max_len, temperature)
        print(f"New Word: {word}")

if __name__ == "__main__":
    main()