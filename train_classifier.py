import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt  # For confusion matrix visualization
import seaborn as sns # For confusion matrix visualization

# Load data (handle potential KeyError as discussed before)
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Filter incomplete data (crucial to avoid ValueError)
complete_indices = []
expected_landmark_count = 21 * 2  # 21 landmarks * 2 coordinates (adjust if needed)
for i, sample in enumerate(data):
    if len(sample) == expected_landmark_count:
        complete_indices.append(i)

filtered_data = [data[i] for i in complete_indices]
filtered_labels = [labels[i] for i in complete_indices]

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)


# Split data (stratify is good practice)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# 1. Hyperparameter Tuning with GridSearchCV (Important!)
param_grid = {
    'n_estimators': [50, 100, 200],  # Test different numbers of trees
    'max_depth': [None, 10, 20],     # Test different tree depths
    'min_samples_split': [2, 5, 10],  # Test different minimum splits
    'min_samples_leaf': [1, 2, 4]      # Test different minimum leaf samples
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1) #n_jobs uses all processors
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_


# 2. Evaluate with More Metrics
y_predict = best_model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'Accuracy: {score * 100:.2f}%')

print(classification_report(y_test, y_predict))  # Precision, recall, F1-score
cm = confusion_matrix(y_test, y_predict)

# 3. Visualize Confusion Matrix (Very Helpful)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(labels), yticklabels=np.unique(labels)) #Use np.unique to avoid issues with labels not being strings
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



# Save the *best* model from the grid search
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)

print("Model saved.")