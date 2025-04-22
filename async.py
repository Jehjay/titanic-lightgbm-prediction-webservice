from fastapi import FastAPI, HTTPException
import uvicorn
import requests
from celery import Celery
import lightgbm as lgb
import pandas as pd
import model
import logging

app = FastAPI()

# Initialize logging
logger = logging.getLogger()
#logger.setLevel(logging.WARNING)
#logging.basicConfig(level=logging.INFO, filename='async.log')
log_handler = logging.FileHandler(filename='async.log', encoding='utf-8')
logging.basicConfig(handlers=[log_handler], level=logging.WARNING)

logger.error('API NOT FUNCTIONING CORRECTLY')

# Celery configuration
celery = Celery('predict_task', broker='redis://localhost:6379/0')

# Load the model
#lightgbm_classifier_model = joblib.load('model.pkl')

lightgbm_classifier_model = lgb.Booster(model_file='model.txt')

@celery.task
def predict_task(data: dict):
    # convert the input data to a Pandas DataFrame
    df = pd.DataFrame([data])

    # get models features in the correct order
    column = lightgbm_classifier_model.feature_name()

    # use column to reindex the prediction dataframe
    df = df.reindex(columns=column)

    # get predictions
    predictions = lightgbm_classifier_model.predict(df)
    return{"predictions":predictions.tolist()}

    # get prediction probability
    #predictions_probability = lightgbm_classifier_model.predict_proba(df)
    #return{"predictions":predictions_probability.tolist()}

    # prediction and probability  
    #prediction_probability = []
    
    #for i in predictions.tolist():
    #   for j in predictions_probability.tolist():
    #       prediction_probability.append('predicted result : ' + str(i) + ' - ' + 'probability : ' + str(j))
    
    #return{"prediction and probability":prediction_probability}

    
@app.post("/titanic_async")
async def predict(data: dict):
    # Enqueue the prediction task
    task = predict_task.delay(data)

    # Return the task ID
    return {"task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    # Retrieve the result of the job
    result = predict_task.AsyncResult(task_id)

    if result.ready():
        return {"predictions": result.get()}
    else:
        return {"status": "pending"}