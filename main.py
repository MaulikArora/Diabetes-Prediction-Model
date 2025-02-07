import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r"D:\VSC\python\ml\Diabetes Prediction\diabetes_dataset.csv", encoding="latin-1")

df = df.join(pd.get_dummies(df.Gender))
df = df.drop("Gender", axis=1)

df = df.join(pd.get_dummies(df.Ethnicity))
df = df.drop("Ethnicity", axis=1)

activity = pd.get_dummies(df["Physical_Activity_Level"])
activity.columns = ["PAL_" + col for col in activity.columns]
df = df.join(activity)
df = df.drop("Physical_Activity_Level", axis=1)

stress = pd.get_dummies(df["Stress_Level"])
stress.columns = ["SL_" + col for col in stress.columns]
df = df.join(stress)
df = df.drop("Stress_Level", axis=1)

smoke = pd.get_dummies(df["Smoking_Status"])
smoke.columns = ["Smk_" + col for col in smoke.columns]
df = df.join(smoke)
df = df.drop("Smoking_Status", axis=1)

print(df.columns)

target = df.corr()["Diabetes_Diagnosis"].apply(abs)

x, y = df.drop("Diabetes_Diagnosis", axis=1), df["Diabetes_Diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

scalar = StandardScaler()

x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)

forest = RandomForestRegressor()
forest.fit(x_train_scaled, y_train)
forest.score(x_test_scaled, y_test)

joblib.dump(forest, "diabetes_model.pkl")
joblib.dump(scalar, "scaler.pkl")
