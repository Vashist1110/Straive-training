import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome",axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12, stratify=y
)

model = DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Training Accuracy: ", model.score(X_train,y_train))
print("Testing accuracy: ",accuracy_score(y_test,y_pred))

plt.figure(figsize=(16,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Diabetes", "Diabetes"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
