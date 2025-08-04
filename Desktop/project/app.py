import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('Iris.csv')
df.drop('Id', axis=1, inplace=True)

# Encode labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-NN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.2f}")

# Optional: classification report
print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, models["Decision Tree"].predict(X_test), target_names=le.classes_))
# Reload dataset for plotting with original labels
df_plot = pd.read_csv('Iris.csv')
sns.pairplot(df_plot, hue='Species')
plt.show()
# Visualize Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(models["Decision Tree"], feature_names=X.columns, class_names=le.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
sns.pairplot()
plot_tree()


