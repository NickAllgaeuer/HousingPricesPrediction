# House Price Prediction using Neural Networks

This project uses a neural network to predict house prices based on a set of features. The dataset used in this project is available in train.csv file.
Required Libraries

    pandas
    numpy
    tensorflow
    matplotlib
    sklearn

```python

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from classes_mapping import class_mapping
```

## Loading and Preprocessing Data

The train.csv file is loaded into a Pandas DataFrame using the read_csv() function. The data is then preprocessed by converting categorical variables to numerical ones and replacing any missing values with -1.

```python

df = pd.read_csv("train.csv", na_filter=False)

def get_inputs_from_df(df):
    # rows using values first
    inputs = []
    labels = ["LotFrontage",
        "LotArea",
        "YearBuilt",
        "YearRemodAdd",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageYrBlt",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "MoSold",
        "YrSold"]
    
    for label in labels:
        df[label] = df[label].replace("NA", -1)
        column = df[label].to_numpy().reshape(-1, 1).astype(np.float64)
        inputs.append(column)
    
    for i, key in enumerate(class_mapping):
        for value in class_mapping[key]:
            inputs.append((df[key] == value).astype(np.float64).to_numpy().reshape(-1, 1) )

    
    inputs = tuple(inputs)
    preprocessed_inputs = np.concatenate(inputs, axis=1)
    
    return preprocessed_inputs 

def get_outputs_from_df(df):
    return df["SalePrice"].to_numpy().reshape(-1, 1).astype(np.float64)

X = get_inputs_from_df(df)
Y = get_outputs_from_df(df)
```

## Splitting Data into Train and Test Sets

The preprocessed data is split into training and testing sets using the train_test_split() function from the sklearn library.

```python

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
```

## Building the Neural Network Model

The neural network model is built using the Sequential model from Keras. It has 4 dense layers, each with ReLU activation, and 2 dropout layers. The last layer has a linear activation function as the target variable is continuous.

```python

dropout_rate = 0.35
model = Sequential([
    Dense(units=16, input_dim=X_train.shape[1], activation="relu", kernel_initializer='normal', ),
    Dense(units=32, activation="relu"),
    Dropout(dropout_rate),
    Dense(units=64, activation="relu"),
    Dropout(dropout_rate),
    Dense(units=128, activation="relu"),
    Dense(units=1, activation="relu")
    ])
```

## Model Training

The model is trained using the Adam optimizer with a learning rate of 0.0001 and mean squared error as the loss function. The model is trained for 300 epochs with a batch size of 10. The training and validation losses are plotted using matplotlib.
Model Evaluation

The model is evaluated using the mean absolute error between the predicted and actual sale prices. The mean absolute error is calculated as a percentage of the mean sale price. The mean validation price and mean validation error are printed to the console.

## Conclusion
In this project, we have trained a neural network model to predict housing prices based on a set of features. We started by importing the necessary libraries and reading in the dataset using pandas. We then preprocessed the data by selecting the relevant features and converting them to numeric values.

We then split the data into training and testing sets using the train_test_split function from scikit-learn. We defined a neural network model with multiple hidden layers and used the mean squared error loss function to train the model. We also used dropout regularization to prevent overfitting.

After training the model, we evaluated its performance on the test set and computed the mean validation error. We plotted the training and validation loss curves to visualize the training process.

Overall, the model achieved a mean validation error of around 17% on the test set, which indicates that it can predict housing prices with reasonable accuracy. However, there is still room for improvement, and we can further optimize the model by tuning hyperparameters, adding more features, or using more sophisticated neural network architectures.

In conclusion, this project demonstrates the power of neural networks in solving regression problems and highlights the importance of proper data preprocessing and model evaluation.
