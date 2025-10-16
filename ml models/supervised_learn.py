from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample dataset: House prices
data = {
    "sqft": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 5],
    "price": [200000, 250000, 300000, 400000, 500000]
}
df = pd.DataFrame(data)

# Features and target
X = df[["sqft", "bedrooms"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
print("Predicted Prices:", pred)
