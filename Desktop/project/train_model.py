import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv('Iris.csv')
df.drop('Id', axis=1, inplace=True)

# Encode labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Features and labels
X = df.drop('Species', axis=1)
y = df['Species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training label distribution:")
print(y_train.value_counts())

# Train a Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoder using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump((model, le), f)

print("Model saved successfully!")
