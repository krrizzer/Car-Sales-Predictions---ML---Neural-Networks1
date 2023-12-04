# Car Purchase Amount Prediction Model

## Overview
This project aims to predict the car purchase amount for individuals based on several features using a deep learning model. The model is built using the Keras library in Python and employs a Sequential neural network for regression tasks. A graphical user interface (GUI) is also provided for easy interaction with the model.

## Dataset
The dataset used for this project, `Car_Purchasing_Data.csv`, contains several features including:
- Gender
- Age
- Annual Salary
- Credit Card Debt
- Net Worth

The target variable is the 'Car Purchase Amount'.

## Data Preprocessing
The data preprocessing steps include:
1. **Loading the Data**: The data is loaded into a Pandas DataFrame.
2. **Feature Selection**: Unnecessary columns like 'Customer Name', 'Customer e-mail', and 'Country' are dropped.
3. **Normalization**: The input features and target variable are normalized using `MinMaxScaler` from the `sklearn.preprocessing` module.

## Model Architecture
The model is a Sequential neural network comprising:
- An input layer with 30 neurons and a 'sigmoid' activation function. The input dimension matches the number of features in the dataset.
- A hidden layer with 30 neurons and a 'relu' activation function.
- An output layer with a single neuron (since this is a regression task) and a 'linear' activation function.

## Model Training
The model is compiled with the 'adam' optimizer and 'mean_squared_error' as the loss function. It is then trained on the dataset for 500 epochs with a batch size of 20, and 20% of the data is used as a validation split.

## Graphical User Interface (GUI)
The GUI, built using the `tkinter` library, allows users to input feature values and get the predicted car purchase amount. It includes entry fields for Gender, Age, Annual Salary, Credit Card Debt, and Net Worth. The GUI uses the trained model and scaler to make predictions based on user input.

## Running the Application
To run the application:
1. Ensure you have Python installed along with required libraries: `pandas`, `numpy`, `sklearn`, `keras`, and `tkinter`.
2. Clone the repository and navigate to the project directory.
3. Run the Python script, which opens the GUI for making predictions.

## Future Scope
- Enhancing the model by experimenting with different architectures and hyperparameters.
- Integrating more features that could impact the car purchase amount.
- Implementing additional validation checks in the GUI for user inputs.
