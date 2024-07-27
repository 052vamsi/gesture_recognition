from collections import Counter
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Check class distribution
class_counts = Counter(labels)
print("Class counts:", class_counts)

# Remove classes with fewer than 2 samples
min_samples = 2
filtered_data = []
filtered_labels = []

for label in class_counts:
    if class_counts[label] >= min_samples:
        filtered_data.extend([data[i] for i in range(len(labels)) if labels[i] == label])
        filtered_labels.extend([labels[i] for i in range(len(labels)) if labels[i] == label])

# Convert to numpy arrays
filtered_data_array = np.array(filtered_data)
filtered_labels_array = np.array(filtered_labels)

print("Filtered data shape:", filtered_data_array.shape)
print("Filtered labels shape:", filtered_labels_array.shape)

# Pad or truncate data entries to ensure uniform length of 42 features
expected_feature_length = 21 * 2  # 21 landmarks, each with x and y coordinates
data_padded = [entry[:expected_feature_length] + [0] * (expected_feature_length - len(entry)) for entry in filtered_data]

data_array = np.array(data_padded)
labels_array = np.array(filtered_labels)

# Print shapes for debugging
print(f"Data shape: {data_array.shape}")
print(f"Labels shape: {labels_array.shape}")

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array)

# Initialize and train the classifier
model = RandomForestClassifier(class_weight='balanced')
model.fit(x_train, y_train)

# Predict on the test set
y_predict = model.predict(x_test)

# Calculate accuracy score
score = accuracy_score(y_test, y_predict)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
