import numpy as np

class SimpleTokenizer:
    def __init__(self):
        self.word_index = {}
        self.index = 1

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_index:
                    self.word_index[word] = self.index
                    self.index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 0) for word in text.lower().split()]
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, maxlen):
    padded = []
    for seq in sequences:
        seq = seq[:maxlen]
        padded.append([0]*(maxlen - len(seq)) + seq)
    return np.array(padded)