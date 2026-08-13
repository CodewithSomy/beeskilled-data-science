import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

maths_df = pd.read_excel(r'Week3_project\Maths.xlsx')

X = maths_df[
    [
        'age',
        'Medu',
        'Fedu',
        'traveltime',
        'studytime',
        'failures',
        'famrel',
        'freetime',
        'goout',
        'Dalc',
        'Walc',
        'health',
        'absences'
    ]
]

y = maths_df['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Made predictions on the test set
y_pred = model.predict(X_test)

# Evaluated the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Student Performance Prediction - Linear Regression")
print("---------------------------------------------------")
print("R² Score:", round(r2, 4))
print("Mean Absolute Error:", round(mae, 4))

results = pd.DataFrame({
    'Actual G3': y_test.values,
    'Predicted G3': y_pred
})

print("\nSample Predictions:")
print(results.head(10))