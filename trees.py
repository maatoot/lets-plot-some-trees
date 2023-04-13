import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


titanic_df = pd.read_csv('titanic-passengers.csv', sep=';')

# data cleaning
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
titanic_df['Sex'] = titanic_df['Sex'].map({'female': 0, 'male': 1})
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'])
titanic_df = titanic_df.dropna()

# split data into training and testing sets
X = titanic_df.drop(['Survived'], axis=1)
y = titanic_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the decision tree
plt.figure(figsize=(10,6))
plot_tree(dt, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.show()

# Train and evaluate a decision tree with different parameters
dt2 = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt2.fit(X_train, y_train)
y_pred2 = dt2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy with different parameters:", accuracy2)

# Train a random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred3 = rf.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred3)
print("Accuracy with random forest:", accuracy3)
