import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(
            self,
            mobile_brand:str,
            mobile_rating:int,
            network_type:int,
            nfc_availability:int,
            ir_blaster_availability:int,
            processor_brand:str,
            num_cores:int,
            processor_speed:float,
            ram:int,
            internal_memory:int,
            battery:int,
            fast_charge_available:int,
            fast_charge_watt:int,
            refresh_rate:int,
            num_rear_camera:int,
            main_rear_camera:int,
            num_front_camera:int,
            main_front_camera:int,
            os:str,
            extended_memory_availability:int,
            ex_mem_capacity:int,
            ppi:int):
        self.mobile_brand=mobile_brand
        self.mobile_rating=mobile_rating
        self.network_type=network_type
        self.nfc_availibility=nfc_availability
        self.ir_blaster_availibility=ir_blaster_availability
        self.processor_brand=processor_brand
        self.num_cores=num_cores
        self.processor_speed=processor_speed
        self.ram=ram
        self.internal_memory=internal_memory
        self.battery=battery
        self.fast_charge_available=fast_charge_available
        self.fast_charge_watt=fast_charge_watt
        self.refresh_rate=refresh_rate
        self.num_rear_camera=num_rear_camera
        self.main_rear_camera=main_rear_camera
        self.num_front_camera=num_front_camera
        self.main_front_camera=main_front_camera
        self.os=os
        self.extended_memory_available=extended_memory_availability
        self.ex_mem_capacity=ex_mem_capacity
        self.ppi=ppi

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "brand_name":[self.mobile_brand],
                "rating":[self.mobile_rating],
                "has_5G":[self.network_type],
                "has_nfc":[self.nfc_availibility],
                "has_ir_blaster":[self.ir_blaster_availibility],
                "processor_brand":[self.processor_brand],
                "num_cores":[self.num_cores],
                "processor_speed":[self.processor_speed],
                "ram_capacity":[self.ram],
                "internal_memory":[self.internal_memory],
                "battery_capacity":[self.battery],
                "fast_charging_available":[self.fast_charge_available],
                "fast_charging":[self.fast_charge_watt],
                "refresh_rate":[self.refresh_rate],
                "num_rear_cameras":[self.num_rear_camera],
                "num_front_cameras":[self.num_front_camera],
                "rear_main_camera":[self.main_rear_camera],
                "front_main_camera":[self.main_front_camera],
                "os":[self.os],
                "extended_memory_available":[self.extended_memory_available],
                "extended_upto":[self.ex_mem_capacity],
                "ppi":[self.ppi]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        