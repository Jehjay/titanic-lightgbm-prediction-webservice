# Titanic Lightgbm Prediction Webservice

<h2> <i>Aim</i> </h2>		
<ul>
<li> To build a web service that accepts input, runs a prediction on it, and returns the predicted result with probability
using the lightgbm model on Titanic data and implementing a synchronous API and an asynchronous API approach.
</ul>

**<h2> Building the webservice </h2>**
<ul>
<li> The standard library’s virtual environment tool venv was used to develop the service in a virtual environment and install packages. </li>
<li> A list of all installed packages and their versions can be found in the requirements.txt file. </li>
<li> Titanic data was obtained from Kaggle. Subsequent steps for loading, preprocessing, analysing and transforming data are carried out in the model.py file </li>
<li> **LightGBM** - was installed to implement the prediction model in the model.py file. In addition, the model was exported to be used in the sync and async API's. </li>
<li> **FastAPI** - was used to set up sync and async endpoints. </li>
<li> **Logging** - was used to track the web service performance. </li>
<li>**Celery** - an open-source asynchronous job queue based on distributed message passing, supports scheduling and focuses on operations in real time, was used to implement the async api using Redis as a message broker to run in the async.py file </li>
<li> The sync api endpoint implementation can be found in the sync.py file </li>
</ul>

<i>Techstack</i>: Python 3.7, numpy, pandas, matplotlib, sklearn, seaborn. </li>


**<h2> Running the web service </h2>**	
<ul>
<li> Uvicorn - was used to serve the web service and containerised in Dockerfile.sync and Dockerfile.async, respectively. To load the server for sync and async endpoints, run the following commands in the console: </li>
```docker
docker run -p 8000:8000 lgb-fastapi-sync-app
```
```python
docker run -p 8000:8000 lgb-fastapi-async-app
```
<li> If everything works as intended, you can see the server running at http://127.0.0.1:8000. </li>
<ul/>

  
**<h2> Using the web service </h2>**
<ul>
<li> After starting your server, you can run the following scripts for sync and async endpoints and provide a JSON input to see if the web service works as intended: </li>
</ul>
```python
sync_testing.py
```
```python
async_testing.py
```

