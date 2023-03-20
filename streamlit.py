import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the data
data = pd.read_csv("gold_prices.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
X = data.index.values.astype(float).reshape(-1, 1)
y = data["Close"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a Streamlit app
st.title("Gold Price Prediction App")
st.write("Enter a date to get the predicted gold price.")
date_input = st.date_input("Date", value=pd.to_datetime("2022-01-01"), min_value=data.index.min(), max_value=data.index.max())
date_value = date_input.timestamp()
price = model.predict([[date_value]])
st.write("Predicted Gold Price:", round(price[0], 2))

# Display model performance metrics
st.write("Model Performance:")
st.write("Mean Squared Error:", round(mse, 2))
st.write("R-squared:", round(r2, 2))
