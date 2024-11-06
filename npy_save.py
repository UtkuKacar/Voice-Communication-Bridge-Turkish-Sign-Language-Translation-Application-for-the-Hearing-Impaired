import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data4')

# Actions that we try to detect
actions = np.array(['Evet', 'Güle Güle', 'Günaydın', 'Hayır', 'Merhaba', 'Özür dilerim', 'Rica Ederim', 'Seni Seviyorum', 'Teşekkür Ederim'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Initialize lists to hold data and labels
X = []
y = []

# Loop through each action
for action in actions:
    for sequence in range(no_sequences):
        sequence_data = []
        for frame_num in range(sequence_length):
            try:
                # Load the keypoints from the .npy file
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy')
                keypoints = np.load(npy_path)
                sequence_data.append(keypoints)
            except FileNotFoundError:
                print(f"File {npy_path} not found. Skipping this file.")
            except Exception as e:
                print(f"An error occurred while loading {npy_path}: {e}")
        
        if len(sequence_data) == sequence_length:
            X.append(sequence_data)
            y.append(action)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

# Ensure y is in the correct shape for categorical cross-entropy
y = tf.keras.utils.to_categorical(y, num_classes=len(actions))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save the data and labels to .npy files
np.save('X_train4.npy', X_train)
np.save('y_train4.npy', y_train)
np.save('X_test4.npy', X_test)
np.save('y_test4.npy', y_test)
np.save('actions4.npy', actions)

print("Training and test data saved successfully as X_train4.npy, y_train4.npy, X_test4.npy, y_test4.npy, and actions4.npy")
print("X_train4 shape:", X_train.shape)
print("y_train4 shape:", y_train.shape)
print("X_test4 shape:", X_test.shape)
print("y_test4 shape:", y_test.shape)
print("actions shape:", actions.shape)
