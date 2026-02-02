import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scale=preprocessor.transform(features)
            preds=model.predict(data_scale)

            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
        self,
        Store:int,
        Holiday_Flag:int,
        Fuel_Price:float,
        CPI:float,
        Unemployment:float,
        Temperature_C:float,
        Year:int,
        Month_Name:str):
        self.Store = Store
        self.Holiday_Flag = Holiday_Flag
        self.Fuel_Price = Fuel_Price
        self.CPI = CPI
        self.Unemployment = Unemployment
        self.Temperature_C = Temperature_C
        self.Year = Year
        self.Month_Name = Month_Name

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Store":[self.Store],
                "Holiday_Flag":[self.Holiday_Flag],
                "Fuel_Price":[self.Fuel_Price],
                "CPI":[self.CPI],
                "Unemployment":[self.Unemployment],
                "Temperature_C":[self.Temperature_C],
                "Year":[self.Year],
                "Month_Name":[self.Month_Name]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)