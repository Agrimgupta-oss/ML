"""ML preprocessing + simple linear regression demo.

This file is generated from the notebook code you shared in chat.
It demonstrates:
- Handling missing values (dropna / fillna)
- Label Encoding
- One Hot Encoding (sklearn + pandas)
- Feature scaling (StandardScaler)
- Train/test split and LinearRegression prediction

Run:
  python3 ml_code.py
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def demo_label_encoding_colors() -> pd.DataFrame:
    data = {
        "colors": ["red", "green", "blue", "orange", "green", "blue", np.nan]
    }
    df = pd.DataFrame(data)

    # handle missing values
    df.dropna(inplace=True)

    # label encoding (multiple methods)
    le = LabelEncoder()
    df["colors_encoder1"] = le.fit_transform(df["colors"])

    df["colors_encoder2"] = LabelEncoder().fit_transform(df["colors"])

    import sklearn

    df["colors_encoder3"] = sklearn.preprocessing.LabelEncoder().fit_transform(df["colors"])

    print("[Label Encoding Colors] after encoding columns:")
    print(df)

    # one hot encoding (sklearn + pandas)
    # Drop cols of label encoding (keep original colors)
    if "colors_encoder" in df.keys():
        df.drop(
            df[
                [
                    "colors_encoder",
                    "colors_encoder1",
                    "colors_encoder2",
                    "colors_encoder3",
                ]
            ],
            axis=1,
            inplace=True,
        )

    print("\n[One Hot Encoding - sklearn] :")
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[["colors"]])
    print(encoded.toarray())

    print("\n[One Hot Encoding - pandas] :")
    df_encoder = pd.get_dummies(df, columns=["colors"])
    print(df_encoder)

    return df


def demo_combination_preprocessing() -> pd.DataFrame:
    # combination of handle missing value | label encoding | onehot encoding
    data = {
        "age": [10, 20, np.nan, 26, 30],
        "gender": ["male", "female", "others", "male", np.nan],
        "name": ["dheeraj", "kavi", "sapu", "yash", "hello"],
    }

    df = pd.DataFrame(data)

    # handle missing values
    df["age"] = df["age"].fillna(df["age"].mean())
    df.dropna(subset=["gender"], inplace=True)
    print("\n[Combination] after missing value handling:")
    print(df)

    # label encoding
    le = LabelEncoder()
    df["gender1"] = le.fit_transform(df["gender"])
    print("\n[Combination] after label encoding:")
    print(df)

    # one hot encoding
    oe = OneHotEncoder()
    oe_e = oe.fit_transform(df[["gender"]]).toarray()
    print("\n[Combination] one hot (sklearn) array:")
    print(oe_e)

    return df


def demo_feature_scaling() -> pd.DataFrame:
    data = {"experience": [300, 600, 900, 1500], "salary": [1000, 1500, 2000, 3000]}
    df = pd.DataFrame(data)
    print("\n[Feature Scaling] original:")
    print(df)

    scaler = StandardScaler()
    df["experience"] = scaler.fit_transform(df[["experience"]])  # 2D

    print("\n[Feature Scaling] scaled experience:")
    print(df)

    # split data
    X = df[["experience"]]
    y = df["salary"]

    print("\nX (experience):")
    print(X)
    print("\ny (salary):")
    print(y)

    return df


def demo_train_test_linear_regression():
    # load dataset
    data_url = (
        "https://raw.githubusercontent.com/rajendra0968jangid/Ds-Arya/main/extended_salary_data.csv"
    )
    df = pd.read_csv(data_url)

    X = df[["experience"]]  # 2D
    y = df["salary"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    # input from user
    user = int(input("Enter your Experience: "))
    new_data = {"experience": [user]}
    df1 = pd.DataFrame(new_data)

    pred_data = model.predict(df1)
    print(pred_data[0])


if __name__ == "__main__":
    # Run small demos (you can comment any sections out)
    demo_label_encoding_colors()
    demo_combination_preprocessing()
    demo_feature_scaling()

    # This one requires user input and trains the model.
    demo_train_test_linear_regression()

