
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [15,20,35,40,50,60,65,75,85,95]
}

df = pd.DataFrame(data)

X = df[["Study_Hours"]]
y = df["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeRegressor(
    max_depth=2, random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Model evaluation
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "MAE": [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, dt_pred)
    ],
    "R2 Score": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, dt_pred)
    ]
})

print("Model Comparison:")
print(results)

# New student predictions
new_students = pd.DataFrame({
    "Study_Hours": [2, 4, 6, 8, 10]
})

print("\nNew Student Predictions:")
print(pd.DataFrame({
    "Study Hours": new_students["Study_Hours"],
    "Predicted Marks": lr_model.predict(new_students)
}).round(2))

# Plot graph
hours = pd.DataFrame({
    "Study_Hours": np.linspace(1, 10, 100)
})

plt.figure(figsize=(9, 5))
plt.scatter(X, y, label="Actual Marks")
plt.plot(
    hours["Study_Hours"],
    lr_model.predict(hours),
    label="Linear Regression"
)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.grid()
plt.savefig("student_marks_graph.png")
plt.show()
