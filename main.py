import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from classes_mapping import class_mapping

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)

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

model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss="mean_squared_error", metrics=["mean_squared_error"])

history = model.fit(x=X_train, y=Y_train, validation_split=0.3, batch_size=10, epochs=200, shuffle=True, verbose=1)
print(f'Trained for {len(history.history["loss"])} epochs.')

history = history.history
fig, ax = plt.subplots()
for key in ["loss", "val_loss"]:
    ax.plot(history[key], label=key)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.title("Loss")
plt.show()

pred = model.predict(x=X_test, verbose=2)
error = np.abs(pred - Y_test)
mean_price = np.mean(Y_test)
mean_error = np.mean(error)

print("Mean Validation Price:", round(mean_price))
print("Mean Validation Error (Absolute):", round(mean_error))
print(f"Mean Validation Error (Relative): {round(mean_error/mean_price*100)}%")

