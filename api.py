import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

model = joblib.load("diabetes_model.pkl")

app = FastAPI()

class PredictionInput(BaseModel):
    Age: float
    Income: float
    BMI: float
    Blood_Pressure: float
    Cholesterol: float
    Exercise_Hours_Per_Week: float
    Alcohol_Consumption_Per_Week: float
    Family_History_Diabetes: int
    Glucose_Level: float
    HbA1c: float
    Insulin_Resistance: float
    Heart_Disease_History: int
    Fast_Food_Intake_Per_Week: float
    Processed_Food_Intake_Per_Week: float
    Daily_Caloric_Intake: float
    Sleep_Hours_Per_Night: float
    Medication_Use: int
    Female: int
    Male: int
    Asian: int
    Black: int
    Hispanic: int
    Other: int
    White: int
    PAL_High: int
    PAL_Low: int
    PAL_Moderate: int
    SL_High: int
    SL_Low: int
    SL_Moderate: int
    Smk_Current: int
    Smk_Former: int
    Smk_Never: int

@app.post("/predict")
def predict(input_data: PredictionInput):
    input_values = [input_data.Age, input_data.Income, input_data.BMI, input_data.Blood_Pressure,
                    input_data.Cholesterol, input_data.Exercise_Hours_Per_Week,
                    input_data.Alcohol_Consumption_Per_Week, input_data.Family_History_Diabetes,
                    input_data.Glucose_Level, input_data.HbA1c, input_data.Insulin_Resistance,
                    input_data.Heart_Disease_History, input_data.Fast_Food_Intake_Per_Week,
                    input_data.Processed_Food_Intake_Per_Week, input_data.Daily_Caloric_Intake,
                    input_data.Sleep_Hours_Per_Night, input_data.Medication_Use, input_data.Female,
                    input_data.Male, input_data.Asian, input_data.Black, input_data.Hispanic,
                    input_data.Other, input_data.White, input_data.PAL_High, input_data.PAL_Low,
                    input_data.PAL_Moderate, input_data.SL_High, input_data.SL_Low, input_data.SL_Moderate,
                    input_data.Smk_Current, input_data.Smk_Former, input_data.Smk_Never]

    input_values = [input_values]

    prediction = model.predict(input_values)

    return {"prediction": prediction[0]}

