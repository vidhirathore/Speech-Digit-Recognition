from hmmlearn import hmm
import numpy as np

class HMMModel:
    def __init__(self, n_mfcc, n_components):
        self.n_mfcc = n_mfcc
        self.n_components = n_components
        self.models = {i: None for i in range(10)}  

    def train(self, train_data, train_labels):
        """
        Train an HMM for each digit using the provided training data.
        """
        for digit in range(10):
            digit_data = [mfcc.T for mfcc, label in zip(train_data, train_labels) if label == digit]
            X = np.concatenate(digit_data)  
            lengths = [len(mfcc) for mfcc in digit_data] 
            model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="diag", n_iter=150)
            model.fit(X, lengths)
            self.models[digit] = model

    def predict(self, test_data):
        """
        Predict the digit for each sample in the test data.
        """
        predictions = []
        for mfcc in test_data:
            mfcc = mfcc.T 
            scores = {digit: model.score(mfcc) for digit, model in self.models.items()}
            predicted_digit = max(scores, key=scores.get)  
            predictions.append(predicted_digit)
        
        return predictions
