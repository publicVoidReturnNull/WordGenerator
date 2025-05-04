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