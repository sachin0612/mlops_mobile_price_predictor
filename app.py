from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app= Flask(__name__)

## Route for a home page

@app.route('/')
def index():
    return render_template('experiment.html')

@app.route('/',methods=['post'])
def predict_datapoint():
    data=CustomData(
        mobile_brand=request.form.get('mobile_brand'),
        mobile_rating=request.form.get('mobile_rating'),
        network_type=request.form.get('network_type'),
        nfc_availability= request.form.get('nfc'),
        ir_blaster_availability=request.form.get('ir'),
        processor_brand=request.form.get('processor_brand'),
        num_cores=request.form.get('cores'),
        processor_speed=request.form.get('processor_speed'),
        ram=request.form.get('ram_capacity'),
        internal_memory=request.form.get('internal_memory'),
        battery=request.form.get('battery_capacity'),
        fast_charge_available=request.form.get('fast_charge_availability'),
        fast_charge_watt=request.form.get('fast_charge'),
        refresh_rate=request.form.get('refresh_rate'),
        num_rear_camera=request.form.get('n_rear_cameras'),
        main_rear_camera=request.form.get('rear_camera_pixel'),
        num_front_camera=request.form.get('n_front_cameras'),
        main_front_camera=request.form.get('front_camera_pixel'),
        os=request.form.get('os'),
        extended_memory_availability=request.form.get('extended_memory_availibility'),
        ex_mem_capacity=request.form.get('extended_memory'),
        ppi=request.form.get('ppi')
    )
    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)
    results=results.round().astype(int)

    return render_template('experiment.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
