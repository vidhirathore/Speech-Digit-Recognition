# import os
# import librosa
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from hmmlearn import hmm
# from sklearn.metrics import accuracy_score

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../models/hmm_scratch'))
# from hmm_scratch import HMMModel

# data_dir = '../../data/external/fsdd/recordings'

# def extract_mfcc(filepath, n_mfcc=20, n_fft=1024):
#     y, sr = librosa.load(filepath, sr=None)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
#     return mfcc.T  # Transpose to get shape (time, n_mfcc)

# def visualize_mfcc(mfcc):
#     """
#     Visualize MFCC as a heatmap.
#     """
#     plt.figure(figsize=(10, 4))
#     sns.heatmap(mfcc.T, cmap='viridis')
#     plt.title("MFCC Heatmap")
#     plt.xlabel("Time")
#     plt.ylabel("MFCC Coefficients")
#     plt.savefig("figures/mfcc_heatmap.png", format="png", dpi=300)
#     plt.show()

# def load_data(data_dir):
#     """
#     Load dataset, extract MFCC features, and separate into training and testing sets.
#     """
#     train_data, test_data = [], []
#     train_labels, test_labels = [], []

#     for filename in os.listdir(data_dir):
#         if filename.endswith(".wav"):
#             digit_label = int(filename.split('_')[0])
#             index = int(filename.split('_')[-1].split('.')[0])

#             filepath = os.path.join(data_dir, filename)
#             mfcc = extract_mfcc(filepath)

#             if index < 5:
#                 test_data.append(mfcc)
#                 test_labels.append(digit_label)
#             else:
#                 train_data.append(mfcc)
#                 train_labels.append(digit_label)

#     return (train_data, train_labels), (test_data, test_labels)

# (train_data, train_labels), (test_data, test_labels) = load_data(data_dir)

# n_mfcc = 20  
# n_components = 5 
# hmm_model = HMMModel(n_mfcc=n_mfcc, n_components=n_components)
# hmm_model.train(train_data, train_labels)

# predictions = hmm_model.predict(test_data)
# accuracy = accuracy_score(test_labels, predictions)
# print(f"Model accuracy: {accuracy * 100:.2f}%")

# def extract_and_predict(filepath, hmm_model):
#     """
#     Extract MFCC features from the custom recorded audio file and predict the digit.
#     """
#     mfcc = extract_mfcc(filepath)
#     predicted_digit = hmm_model.predict([mfcc])[0]
#     return predicted_digit

# custom_audio_dir = '../../data/external/own_voice'
# for digit in range(10):
#     audio_filepath = os.path.join(custom_audio_dir, f"{digit}.wav") 
    
#     if os.path.exists(audio_filepath):
#         predicted_digit = extract_and_predict(audio_filepath, hmm_model)
#         print(f"Predicted digit for {digit} clip: {predicted_digit}")
#     else:
#         print(f"File for digit {digit} not found.")




# # def train_digit_hmm(digit, train_data, train_labels):
# #     """
# #     Train an HMM for a specific digit.
# #     """
# #     digit_data = [mfcc.T for mfcc, label in zip(train_data, train_labels) if label == digit]
# #     X = np.concatenate(digit_data)
# #     lengths = [len(mfcc) for mfcc in digit_data]

# #     model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
# #     model.fit(X, lengths)
# #     return model

# # def predict_digit(mfcc, models):
# #     """
# #     Predict the digit label for a given MFCC sample by scoring it across HMMs.
# #     """
# #     mfcc = mfcc.T 
# #     scores = {digit: model.score(mfcc) for digit, model in models.items()}
# #     return max(scores, key=scores.get)

import os
import librosa
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models/hmm_scratch'))
from hmm_scratch import HMMModel

data_dir = '../../data/external/fsdd/recordings'

def extract_mfcc(filepath, n_mfcc=20, n_fft=1024):
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    return mfcc

def visualize_mfcc(mfcc):
    """
    Visualize MFCC as a heatmap.
    """
    plt.figure(figsize=(10, 4))
    sns.heatmap(mfcc, cmap='viridis')
    plt.title("MFCC Heatmap")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.savefig("figures/mfcc_heatmap.png", format="png", dpi=300)
    plt.show()

def load_data(data_dir):

    train_data, test_data = [], []
    train_labels, test_labels = [], []

    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            parts = filename.split('_')  
            digit_label = int(parts[0]) 
            speaker_name = parts[1] 
            sample_number = int(parts[2].split('.')[0])  
            filepath = os.path.join(data_dir, filename)
            mfcc = extract_mfcc(filepath) 
            if sample_number < 5: 
                test_data.append(mfcc)
                test_labels.append(digit_label)
            else:  
                train_data.append(mfcc)
                train_labels.append(digit_label)

    return (train_data, train_labels), (test_data, test_labels)


(train_data, train_labels), (test_data, test_labels) = load_data(data_dir)

n_mfcc = 20  
n_components = 5
hmm_model = HMMModel(n_mfcc=n_mfcc, n_components=n_components)
hmm_model.train(train_data, train_labels)

predictions = hmm_model.predict(test_data)
accuracy = accuracy_score(test_labels, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

def extract_and_predict(filepath, hmm_model):
    """
    Extract MFCC features from the custom recorded audio file and predict the digit.
    """
    mfcc = extract_mfcc(filepath)
    predicted_digit = hmm_model.predict([mfcc])[0]
    return predicted_digit

custom_audio_dir = '../../data/external/own_voice'
for digit in range(10):
    audio_filepath = os.path.join(custom_audio_dir, f"{digit}.wav") 
    
    if os.path.exists(audio_filepath):
        predicted_digit = extract_and_predict(audio_filepath, hmm_model)
        print(f"Predicted digit for {digit} clip: {predicted_digit}")
    else:
        print(f"File for digit {digit} not found.")
