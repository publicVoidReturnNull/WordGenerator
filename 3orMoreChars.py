"""do generate names but instead of keeping track of prev letter, do prev 2,3,4... letters"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

#helper function
def processSets(startIndex, endIndex, numChars, words):
    inp = []
    label = []
    #create the inputs and labels datasets for inputted sets (thru indices)
    for word in words[startIndex:endIndex]:
        word = ''.join(list('.' for _ in range(numChars))) + word.strip()
        for i in range(len(word)-numChars+1):
            inp.append(''.join(list(word[i: i+numChars])))
            label.append(word[i+numChars] if i+numChars+1 <= len(word) else '.')

    #change from char to int, allow nn to use data
    d = {chr(i+97): i+1 for i in range(26)}
    d['.'] = 0

    inputs = []
    labels = []
    #converts from char to int 
    for i in range(len(inp)):
        inputs.append(list(d[j] for j in inp[i]))
        labels.append(d[label[i]])
    #casts from list to tensor
    inputs = torch.tensor(inputs).float()
    labels = torch.tensor(labels)

    return inputs, labels

def processName(fileName, numChars):
    #read the file
    with open(fileName) as f:
        words = f.readlines()
    #shuffle words
    random.shuffle(words)

    train_index = (int)(len(words) * 0.8)
    val_index = train_index + (int)(len(words) * 0.1)
    test_index = len(words)

    train_input, train_labels = processSets(0, train_index, numChars, words)
    val_input, val_labels = processSets(train_index, val_index, numChars, words)
    test_input, test_labels = processSets(val_index, test_index, numChars, words)
    

    return train_input, train_labels, val_input, val_labels, test_input, test_labels

def runNN(w1, b1, w2, b2, inputs, labels, epochs, batch_size, hiddenLayerOutputs, outputs, val):

    parameters = [w1, b1, w2, b2]

    stepi = []
    lossi = []

    lri = []
    lre = torch.linspace(-3, 0, epochs)
    lrs = 10**lre     #-> used this to find optimal learning_rate
  
    for epoch in range(epochs):
        #creates random batches of size 64
        random_indices = np.random.choice(inputs.shape[0], size=batch_size, replace=False)
        batch = [inputs[r] for r in random_indices]
        batch = torch.stack(batch)
    
        batch_labels = [labels[r] for r in random_indices]
        batch_labels = torch.stack(batch_labels)
        
        #matrix multiplication + a tanh activation
        firstOut = batch @ w1 + b1
        firstOut = torch.tanh(firstOut)

        finalOut = firstOut @ w2 + b2
      
        #calc loss using classification
        loss = torch.nn.functional.cross_entropy(finalOut, batch_labels)
        
        #reset grad
        for p in parameters:
            p.grad = None
        #call backwards on loss (calcs gradient on each param)
        loss.backward()

        #sets learning rate and adjusts gradients
        if val == True:
            learning_rate = lrs[epoch]
            lri.append(learning_rate)
        #hardcoded after finding it was approx most optimal from graphing
        if val == False:
            learning_rate = 0.05 if epoch < 5000 else 0.005
        w1.data += -learning_rate * w1.grad  
        b1.data += -learning_rate * b1.grad 
        w2.data += -learning_rate * w2.grad 
        b2.data += -learning_rate * b2.grad 

        stepi.append(epoch)
        lossi.append(loss.item())
        # print(f"Epoch {epoch+1}, Loss: {loss}")
    return stepi, lossi, lri

def generate_name(w1, b1, w2, b2, num_chars, max_length):
    #conversions from char to int, int to char
    d = {chr(i+97): i+1 for i in range(26)}
    d['.'] = 0
    reverse_d = {v: k for k, v in d.items()}
    
    #padd the begin with ...
    context = [0] * num_chars  
    name = []
    
    for _ in range(max_length):
        input_tensor = torch.tensor([context], dtype=torch.float)

        with torch.no_grad():
            h = torch.tanh(input_tensor @ w1 + b1)
            logits = h @ w2 + b2
            probs = torch.softmax(logits, dim=1)
            #sample next character
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            if next_char_idx == 0:  #stop if '.' is next char 
                break
            name.append(reverse_d[next_char_idx])
            
            context = context[1:] + [next_char_idx]
    
    return ''.join(name)

def main():
    numChars = 4
    fileName = 'MLProj/MYNN/wordgenerate/names.txt'
    train_input, train_labels, val_input, val_labels, test_input, test_labels = processName(fileName, numChars)
    
    batch_size = 32
    hiddenLayerOutputs = 10
    outputs = 27

    #initialize weights and biases of NN
    w1 = torch.randn((len(train_input[0]), hiddenLayerOutputs), requires_grad=True)
    b1 = torch.randn(hiddenLayerOutputs, requires_grad=True)
    w2 = torch.randn((hiddenLayerOutputs, outputs), requires_grad=True)
    b2 = torch.randn(outputs, requires_grad=True)
    
    #train set
    epochs = 1000
    stepi_train, lossi_train, lri_train = runNN(w1, b1, w2, b2, train_input, train_labels, epochs, batch_size, hiddenLayerOutputs, outputs, val=False)
    

    #for validation set
    epochs = 500
    stepi_val, lossi_val, lri_val = runNN(w1, b1, w2, b2, val_input, val_labels, epochs, batch_size, hiddenLayerOutputs, outputs, val=True)
    
    #test set
    epochs = 100
    stepi_test, lossi_test, lri_test = runNN(w1, b1, w2, b2, test_input, test_labels, epochs, batch_size, hiddenLayerOutputs, outputs, val=False)


    #plots the train, val, and test splits
    plt.subplot(1, 3, 1)
    plt.plot(stepi_train, lossi_train)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(lri_val, lossi_val)
    plt.title('Validation Loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(stepi_test, lossi_test)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

    for _ in range(25):  #generate number of names
        print(generate_name(w1, b1, w2, b2, numChars, max_length=15))

if __name__ == "__main__":
    main()

