WordGenerate.py
PURPOSE
The goal is to generate new words that resemble English words but do not exist in the provided dictionary. 
The Markov model captures the likelihood of one character following another (e.g., 'q' often followed by 'u') and uses this to produce sequences of characters, 
ensuring they start and end appropriately.

FEATURES
- Data Processing: Converts words from the NLTK words corpus into integer sequences, mapping a–z to 1–26 (case-insensitive) and using a single 0 token to represent both word start and end.
- Markov Model: Builds a 27x27 transition matrix capturing the probability of transitioning from one character to another, including start/end boundaries.
- Word Generation: Generates new words by sampling characters based on transition probabilities, with a configurable temperature parameter to control randomness.
- Customizable: Allows adjustment of maximum word length and temperature for varied output

REQUIREMENTS
- Python 3.6+
- NLTK (pip install nltk)
- NumPy (pip install numpy)

generateNames.py
PURPOSE
The model processes a list of names from a text file, treating each name as a sequence of characters. 
It learns to predict the next character given the current character, using a bigram approach (i.e., only considering one character of context). 
The training data is encoded as one-hot vectors, and the model parameters (weights) are updated using PyTorch's autograd system.

FEATURES
- Input: A text file (names.txt) containing one name per line.
- Model: A 27x27 weight matrix mapping 27 possible characters (26 lowercase letters + a special . token) to probabilities over the next character.
- Loss: Negative log-likelihood loss (classification-based, not regression).
- Optimization: Gradient descent with a user-defined learning rate.

REQUIREMENTS
- Python 3.6 or higher
- PyTorch (torch)
- A text file (names.txt) with names, one per line

3orMoreChars.py
PURPOSE
WordGenerator is a Python-based neural network project that generates names by predicting the next character based on a sequence of previous characters (e.g., 2, 3, or 4 characters). The model uses a multi-layer perceptron (MLP) with a single hidden layer, trained on a dataset of names (names.txt). It employs PyTorch for tensor operations and matplotlib for visualizing training, validation, and test loss curves.
The project is inspired by character-level language modeling, extending the idea of bigram models to n-grams (e.g., trigrams, 4-grams) to capture more context when generating names.

REQUIREMENTS
- Python 3.6 or higher
- PyTorch (torch)
- A text file (names.txt) with names, one per line

FEATURES
- Customizable Context Length: Generate names using the previous numChars characters (e.g., 2, 3, 4, or more).
- Neural Network: A simple MLP with a configurable hidden layer size, trained using cross-entropy loss and gradient descent.
- Dataset Splitting: Splits the input dataset into training (80%), validation (10%), and test (10%) sets.
- Name Generation: Produces new names up to a maximum length (default: 15 characters) using the trained model.
- Loss Visualization: Plots training, validation, and test loss curves to analyze model performance.

NEXT STEPS
- Resolve the main issue of overfitting by implementing new techniques that capture relationships between letters, such as embeddings
