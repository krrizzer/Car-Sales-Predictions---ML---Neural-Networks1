

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import tkinter as tk

carDB = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

# Check if Gender is already numeric (0 and 1), no need for one-hot encoding
# carDB = pd.get_dummies(carDB, columns=['Gender'])

# Delete unnecessary columns
X = carDB.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)  # Input features
Y = carDB['Car Purchase Amount']  # Target variable

# Normalize the data
scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

scaler_Y = MinMaxScaler()
Y_normalized = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

# Save the scalers
joblib.dump(scaler_X, 'scaler_X.gz')
joblib.dump(scaler_Y, 'scaler_Y.gz')

# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2)

# Build the model
model = Sequential()
model.add(Dense(30, input_dim=X_train.shape[1], activation='sigmoid'))  # Input layer with the number of features as input_dim
model.add(Dense(30, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='linear'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
epochs_hist = model.fit(X_train, Y_train, epochs=500, batch_size=20, verbose=1, validation_split=0.2)

# Save the trained model
model.save('Car_Purchase_Model.h5')

# ... Rest of the GUI code remains the same

# ... [The beginning of the code remains unchanged, including imports, data loading, preprocessing, and model training]

# Save the trained model and scaler
model.save('Car_Purchase_Model.h5')
joblib.dump(scaler_X, 'scaler_X.gz')
joblib.dump(scaler_Y, 'scaler_Y.gz')

# GUI for making predictions with the trained model
# Load the trained model and scalers
model = load_model('Car_Purchase_Model.h5')
scaler_X = joblib.load('scaler_X.gz')
scaler_Y = joblib.load('scaler_Y.gz')

# Function to make prediction based on user input from the GUI
def make_prediction():
    # Retrieve values from GUI
    # Inside your make_prediction function in the GUI
    gender = int(entry_gender.get())
    age = float(entry_age.get())
    salary = float(entry_salary.get())
    debt = float(entry_debt.get())
    net_worth = float(entry_networth.get())

    # Make sure this array structure matches the structure of 'X' used in training
    input_array = np.array([[gender, age, salary, debt, net_worth]])


    # Check if the number of features in input_array matches the expected number
    if input_array.shape[1] != X_train.shape[1]:
        label_result.config(text="Error: Incorrect number of input features.")
        return

    # Normalize the data using the loaded scaler for X
    input_normalized = scaler_X.transform(input_array)

    # Make prediction
    prediction_normalized = model.predict(input_normalized)

    # Reverse normalization using the loaded scaler for Y to get the actual predicted value
    prediction = scaler_Y.inverse_transform(prediction_normalized)

    # Show prediction in GUI
    label_result.config(text=f"Predicted Purchase Amount: ${prediction[0][0]:,.2f}")

# Create the main window
root = tk.Tk()
root.title("Car Purchase Prediction")

# Create and place the input fields
tk.Label(root, text="Enter Gender (0 for Female, 1 for Male):").grid(row=0, column=0)
entry_gender = tk.Entry(root)
entry_gender.grid(row=0, column=1)

tk.Label(root, text="Enter Age:").grid(row=1, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=1, column=1)

tk.Label(root, text="Enter Annual Salary:").grid(row=2, column=0)
entry_salary = tk.Entry(root)
entry_salary.grid(row=2, column=1)

tk.Label(root, text="Enter Credit Card Debt:").grid(row=3, column=0)
entry_debt = tk.Entry(root)
entry_debt.grid(row=3, column=1)

tk.Label(root, text="Enter Net Worth:").grid(row=4, column=0)
entry_networth = tk.Entry(root)
entry_networth.grid(row=4, column=1)

# Button to make the prediction
button_predict = tk.Button(root, text="Predict", command=make_prediction)
button_predict.grid(row=5, column=0, columnspan=2)

# Label to display the result
label_result = tk.Label(root, text="Prediction will show here...")
label_result.grid(row=6, column=0, columnspan=2)

# Run the application
root.mainloop()
