from fastapi import FastAPI, HTTPException
import uvicorn
import requests
import lightgbm as lgb
import pandas as pd
import model
import logging

app = FastAPI()

# Initialize logging
logger = logging.getLogger()
#logger.setLevel(logging.WARNING)
#logging.basicConfig(level=logging.INFO, filename='sync.log')
log_handler = logging.FileHandler(filename='sync.log', encoding='utf-8')
logging.basicConfig(handlers=[log_handler], level=logging.WARNING)

logger.error('API NOT FUNCTIONING CORRECTLY')

# Load the model
#lightgbm_classifier_model = joblib.load('model.pkl')

lightgbm_classifier_model = lgb.Booster(model_file='model.txt')

@app.post("/titanic_sync")
def predict(data : dict):
	# convert the input data to a Pandas DataFrame
	df = pd.DataFrame([data])

	# get models features in the correct order
	column = lightgbm_classifier_model.feature_name()

	# usecolumn to reindex the prediction dataframe
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
	#	for j in predictions_probability.tolist():
	#		prediction_probability.append('predicted result : ' + str(i) + ' - ' + 'probability : ' + str(j))
    
	#return{"prediction and probability":prediction_probability}