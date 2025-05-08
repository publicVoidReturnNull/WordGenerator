import torch

"""DO SAME AS WORDGENERATE, EXCEPT DONT PROCESS WORDS.
use pytorch to keep track of grads, loop thru dataset, store the
total number of times for each of the 26 following characters
normalize to a probability, calculuate loss (or avg loss) not using regression,
but with classification (-log), then with the grad from torch,
change the parameters"""

def processName(fileName):
    """OPENS FILE, THEN SPLITS EACH WORD INTO A LIST OF TUPLES OF 2 CHARACTERS OF THE STRING"""
    with open(fileName) as f:
        words = f.readlines()

    seq = []
    for word in words:
        word = word.strip()
        seq.append([('.', word[0])] + list(zip(word, word[1:])) + [(word[-1],'.')])

    """CREATES THE CONVERSIONS FROM CHAR TO INT AND INT TO CHAR"""
    d = {chr(i+97): i+1 for i in range(26)}
    d['.'] = 0

    dReverse = {i+1: chr(i+97) for i in range(26)}
    dReverse[0] = '.'

    """WILL BE THE DATA USED FOR TRAINING AND TESTING. x[i] -> y[i], MEANING, WHAT COMES AFTER x[i] IS EXPECTED TO BE y[i] eg: for the name emma, what comes after e is expected m"""
    x_vals = []
    y_vals = []
    for s in seq:
        for x,y in s:
            x_vals.append(d[x])
            y_vals.append(d[y])
    x_vals = torch.tensor(x_vals)
    y_vals = torch.tensor(y_vals)

    return x_vals, y_vals, seq, d, dReverse

def main():
    inputs, labels, tupleNames, dictionary, reverseDictionary = processName('MLProj/MYNN/wordgenerate/names.txt')

    oneHot = torch.nn.functional.one_hot(inputs, num_classes=27).float()

    weights = torch.randn((27,27), requires_grad=True)
    
    #you want the onehot@weights[inputs][labels] = 1 or as close as possible to 1 (referring to percentage, so have to normalize by like softmax
    
    epochs = 100
    learning_rate = 50
    loss = 0
    
    for epoch in range(epochs):
    #forward pass + softmax activ
        out = oneHot @ weights
        counts = out.exp()
        prob = counts / counts.sum(1, keepdim=True)

        #calcs loss
        loss = -torch.log(prob[range(len(inputs)), labels]).mean()

        #backward pass and weight update
        weights.grad = None  
        loss.backward()  
        weights.data += -learning_rate * weights.grad  

        print(f"Epoch {epoch+1}, Loss: {loss}")
    

if __name__ == "__main__":
    main()