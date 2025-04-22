# Titanic lightGMB Prediction

# IMPORT LIBRARIES
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date, datetime
# %matplotlib inline
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore")

# LOAD DATA 
FILE_PATH = pathlib.Path.cwd().joinpath('./data/train.csv')
df = pd.read_csv(FILE_PATH)

# PRE-PROCESSING
df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
df_column_names = list(df.columns.values)
print('\ndataframe column names :')
print(df_column_names)

# FILL MISSING VALUES WITH UNKNOWNS
df['age'] = df['age'].fillna('unknown')
#print('age value counts')
#print(df['age'].value_counts())

df['cabin'] = df['cabin'].fillna('unknown')
#print('cabin value counts ')
#print(df['cabin'].value_counts())

df['embarked'] = df['embarked'].fillna('unknown')
#print('embarked value counts')
#print(df['embarked'].value_counts())

# CREATE A FUNCTION TO REMOVE NaN
def remove_nan(x):
    if x == 'unknown':
        return 0
    else:
        return x

df['age'] = df['age'].apply(remove_nan)
df['cabin'] = df['cabin'].apply(remove_nan)
df['embarked'] = df['embarked'].apply(remove_nan)

#print('the dataframe')
#print(df)

# EXPLORATORY DATA ANALYSIS

# UNDERSTANDING THE DATA
#print("\n" + "\n" + "-------------DATAFRAME DESCRIPTION-------------")
#print(round(df[['age','fare']].describe(), 3))
#print("------------------------------------------------------------")

data_analysis = """
age
1. The Average age on the Titanic was 23.799
2. 25% of the passengers were above the age of 35
3. The oldest age for a passenger was 80
fare
1. The Average ticket fare cost was $32.204
2. 25% of the passengers spent more than $31.000
3. The most expensive ticket fare cost was $512.329
"""

print('\ndata_analysis :')
print(data_analysis)

# TRANSFORMATION 
# CREATE A FUNCTION TO ENCODE age
def convert_age_range (val):
    if (0<=val<=12) : return 1
    if (13<=val<=19) : return 2
    if (20<=val<=35) : return 3
    if (36<=val<=50) : return 4
    if val>50 : return 5
    else: return np.nan

df['age'] = df['age'].apply(convert_age_range)

# CREATE A FUNCTION TO ENCODE fare
def convert_fare_range (val):
    if (0<=val<=10) : return 1
    if (10<val<=15) : return 2
    if (15<val<=30) : return 3
    if (30<val<=50) : return 4
    if (val>50) : return 5
    else: return np.nan

df['fare'] = df['fare'].apply(convert_fare_range)

#print('age column')
#print(df.age)

#print('fare column')
#print(df.fare)

# CREATE A FUNCTION FOR ONE-HOT ENCODING THE sex
def sex_one_hot(val):
    if val == 'Female' or val == 'female':
        val = 0
    elif val == 'Male' or val == 'male':
        val = 1
    return val

df['sex'] = df['sex'].apply(sex_one_hot)

#print('sex column')
#print(df.sex)

#CREATE A FUNCTION TO ENCONDE embarked
def convert_embarked_range (val):
    if (val=='S') : return 1
    if (val=='C') : return 2
    if (val=='Q') : return 3
    else: return val

df['embarked'] = df['embarked'].apply(convert_embarked_range)

#print('embarked column')
#print(df.embarked)

#print('the dataframe')
#print(df)

#print('data type count')
#dtypeCount =[df.iloc[:,i].apply(type).value_counts() for i in range(df.shape[1])]
#print(dtypeCount)

# CREATE TARGET AND EXPLANATORY VARIABLES
X = df.drop('survived', axis = 'columns')
y = df['survived']

def titanic_preprocess(X):
    # columns to drop
    to_drop = ['passengerid', 'name', 'ticket', 'cabin', 'embarked']
    
    X_trans = X.drop(to_drop, axis='columns')
    return X_trans

X = titanic_preprocess(X)
df = pd.concat([y,X], axis = 'columns')
#print('the transformed dataframe')
#df.head()

#print('feature variables - X')
#print(X)

#print('target variable - y')
#print(y)

# CREATE TRAIN TEST SPLIT
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X,y)
print('\nTrain Test Split :')
print(len(X_train), len(X_test), len(y_train), len(y_test))
#print('Length of training X data {:.4f}'.format(len(X_train)))
#print('Length of testing  X data {:.4f}'.format(len(X_test)))
#print('Length of training y data {:.4f}'.format(len(y_train)))
#print('Length of training y data {:.4f}'.format(len(y_test)))


# LIGHTGBM MODEL
# training a lightgbm model
print('\ntraining a lightgbm model :')
#lightgbm_classifier_model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
lightgbm_classifier_model = lgb.LGBMClassifier(task='predict',application='binary',learning_rate=0.09,max_depth=-5,early_stopping_round=20,metrics='binary_logloss')
lightgbm_classifier_model.fit(X_train, y_train,eval_set=[(X_test,y_test),(X_train,y_train)], eval_metric='logloss')

# EXPORTING USING lightgbm save_model FOR USE IN sync and async API's
lightgbm_classifier_model.booster_.save_model('model.txt')

# predict the results
lightgbm_y_predict = lightgbm_classifier_model.predict(X_test)
#print('\nLIGHTGBM Model prediction : ')
#print(lightgbm_y_predict)

# predict probability
lightgbm_y_predict_probability = lightgbm_classifier_model.predict_proba(X_test)
#print('\nLIGHTGBM Model predict probability : ')
#print(lightgbm_y_predict_probability)


# prediction and probability  
prediction_probability = []
print('prediction and probability')
for i in lightgbm_y_predict.tolist():
    for j in lightgbm_y_predict_probability.tolist():
#        print('predicted result',  ':',  'probability')
#        print(i, '               :    ', j)
        #print('predicted result : ', i, '-', 'probability : ', j)
        prediction_probability.append('predicted result : ' + str(i) + ' - ' + 'probability : ' + str(j))

#print(prediction_probability)

#[('predicted result : ' + str(i) + ' - ' + 'probability : ' + str(j)) for i in lightgbm_y_predict.tolist() for j in lightgbm_y_predict_probability.tolist()]    
    
# model accuracy
lightgbm_model_accuracy = accuracy_score(y_test, lightgbm_y_predict)
print('\nLIGHTGBM Model Accuracy Score : % s'%(lightgbm_model_accuracy))

# check for overfitting

# model performance on training data
lightgbm_y_predict_train = lightgbm_classifier_model.predict(X_train)

train_set_accuracy_score = accuracy_score(y_train, lightgbm_y_predict_train)
#print("Training set Accuracy Score : % s"%(train_set_accuracy_score))

# training set score
lightgbm_training_set_score = lightgbm_classifier_model.score(X_train, y_train)
#print(' Training set score {:.4f}'.format(lightgbm_training_set_score))

# testing set score
lightgbm_testing_set_score = lightgbm_classifier_model.score(X_test, y_test)
#print(' Testing set score {:.4f}'.format(lightgbm_testing_set_score))

accuracy_analysis = """
There is no significant difference in the training and testing accuracy values.
Therefore, the model has made a significantly accurate prediction.
"""    

print('\naccuracy_analysis :')
print(accuracy_analysis)

# plotting the various feature importance
lgb.plot_importance(lightgbm_classifier_model)
copyright = "\u00A9" + " " + str(date.today().year) + " " + "JM Consulting"
plt.text(0.5, 0.5, copyright, alpha=0.3, fontsize=25, rotation=0, ha='center', va='center', transform=plt.gca().transAxes)
plt.savefig('output/feature_importance.png')

# plotting the metric evaluation
lgb.plot_metric(lightgbm_classifier_model)
copyright = "\u00A9" + " " + str(date.today().year) + " " + "JM Consulting"
plt.text(0.5, 0.5, copyright, alpha=0.3, fontsize=25, rotation=25, ha='center', va='center', transform=plt.gca().transAxes)
plt.savefig('output/metric_evaluation.png')

# plotting the tree
lgb.plot_tree(lightgbm_classifier_model,figsize=(30,40))
copyright = "\u00A9" + " " + str(date.today().year) + " " + "JM Consulting"
plt.text(0.5, 0.5, copyright, alpha=0.3, fontsize=25, rotation=25, ha='center', va='center', transform=plt.gca().transAxes)
plt.savefig('output/tree.png')

#create a confusion matrix
lightgbm_cx_matrix = confusion_matrix(y_test, lightgbm_y_predict)
print('\nconfusion matrix :')
print(lightgbm_cx_matrix)
#print("\ntrue positive (TP) =", lightgbm_cx_matrix[0,0])
#print("\ntrue negative (FP) =", lightgbm_cx_matrix[0,1])
#print("\ntrue negative (FN) =", lightgbm_cx_matrix[1,0])
#print("\ntrue negative (TN) =", lightgbm_cx_matrix[1,1])

#print classification report
lightgbm_classfx_report = classification_report(y_test, lightgbm_y_predict)
print('\nLightGBM Classification Report : ')
print(lightgbm_classfx_report)

#plot the confusion matrix
#lightgbm_cx_matrix_df = pd.DataFrame(data=lightgbm_cx_matrix, columns=['not survived', 'survived'], index=['not survived', 'survived'])
fig, ax = plt.subplots(figsize=(9,6))
sns.heatmap(lightgbm_cx_matrix, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
plt.ylabel('Prediction', fontsize=12)
plt.xlabel('Actual', fontsize=12)
plt.title('LightGBM Confusion Matrix', fontsize=12)
copyright = "\u00A9" + " " + str(date.today().year) + " " + "JM Consulting"
plt.text(0.5, 0.5, copyright, alpha=0.3, fontsize=25, rotation=25, ha='center', va='center', transform=plt.gca().transAxes)
plt.show()
plt.savefig('output/lightgbm_confusion_matrix.png')
#ax.figure.savefig('output/lightgbm_confusion_matrix.png')