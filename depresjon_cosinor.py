
!pip install pyActigraphy

import os
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import pyActigraphy
from pyActigraphy.io import BaseRaw



# Define the path to your uploaded zip file
zip_path = '/content/depresjon.zip'  # Make sure to replace 'your_zip_file.zip' with your actual file name

# Define the directory where you want to extract the files
extracted_path = '/content/extracted/'

# Create the directory if it doesn't exist
os.makedirs(extracted_path, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# List the files in the extracted directory
extracted_files = os.listdir(extracted_path)

# Print the list of files
print("Extracted Files:")
for file in extracted_files:
    print(file)


path = '/content/extracted/data/control/'
dirs = os.listdir(path)
cosinor_result = []
from pyActigraphy.io import BaseRaw
import plotly.graph_objects as go
from pyActigraphy.analysis import Cosinor

print(dirs)
for filepath in dirs:
    source = filepath.split('.')[0]
    if filepath.endswith('.csv'):
      extracted_data = pd.read_csv(path + filepath)
      d = {'timestamp': extracted_data['timestamp'], 'date': extracted_data['date'], 'activity': extracted_data['activity']}
      index = pd.date_range(start=extracted_data['timestamp'].min(),freq='60s',periods=len(extracted_data))
      data = pd.DataFrame(data=d, index=index)
      data['timestamp'] = pd.to_datetime(data['timestamp'])
      #data.set_index('timestamp', inplace=True)
      data['activity'] = extracted_data['activity'].values
      data['timestamp'] = extracted_data['timestamp'].values
      data['date'] = extracted_data['date'].values

      raw = BaseRaw(
            name=source,
            uuid='DeviceId',
            format='Pandas',
            axial_mode=None,
            start_time=data.index[0],
            period=(data.index[-1]-data.index[0]),
            frequency=index.freq,
            data=data['activity'],
            light=None
      )
      cosinor = Cosinor()
      cosinor.fit_initial_params['Period'].vary = False
      results = cosinor.fit(raw.data, verbose=False) # Set verbose to True to print the fit output
      loop_result = results.params.valuesdict()
      loop_result['File_Name'] = raw.name
      cosinor_result.append(loop_result)
df_controls = pd.DataFrame(cosinor_result)
print(df_cosinor_result)




path = '/content/extracted/data/condition/'
dirs = os.listdir(path)
cosinor_result = []
from pyActigraphy.io import BaseRaw
import plotly.graph_objects as go
from pyActigraphy.analysis import Cosinor

print(dirs)
for filepath in dirs:
    source = filepath.split('.')[0]
    if filepath.endswith('.csv'):
      extracted_data = pd.read_csv(path + filepath)
      d = {'timestamp': extracted_data['timestamp'], 'date': extracted_data['date'], 'activity': extracted_data['activity']}
      index = pd.date_range(start=extracted_data['timestamp'].min(),freq='60s',periods=len(extracted_data))
      data = pd.DataFrame(data=d, index=index)
      data['timestamp'] = pd.to_datetime(data['timestamp'])
      #data.set_index('timestamp', inplace=True)
      data['activity'] = extracted_data['activity'].values
      data['timestamp'] = extracted_data['timestamp'].values
      data['date'] = extracted_data['date'].values

      raw = BaseRaw(
            name=source,
            uuid='DeviceId',
            format='Pandas',
            axial_mode=None,
            start_time=data.index[0],
            period=(data.index[-1]-data.index[0]),
            frequency=index.freq,
            data=data['activity'],
            light=None
      )
      cosinor = Cosinor()
      cosinor.fit_initial_params['Period'].vary = False
      results = cosinor.fit(raw.data, verbose=False) # Set verbose to True to print the fit output
      loop_result = results.params.valuesdict()
      loop_result['File_Name'] = raw.name
      cosinor_result.append(loop_result)
df_conditions = pd.DataFrame(cosinor_result)
print(df_conditions)



result_concat_rows = pd.concat([df_controls, df_conditions], axis=0)
print(result_concat_rows)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assuming your DataFrame is named df
# 'label' is the column indicating whether it's a control (0) or condition (1) instance
# Drop 'File_Name' and 'label' from X as they are not features for the model
# Create a binary label column (1 for 'condition', 0 for 'control')
result_concat_rows['LABEL'] = result_concat_rows['File_Name'].str.contains('condition').astype(int)

# Features are all columns except 'FILE_NAME' and the label column
X = result_concat_rows.drop(['File_Name', 'LABEL'], axis=1)

# Target variable is the 'LABEL' column
y = result_concat_rows['LABEL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predictions
linear_predictions = linear_model.predict(X_test)
logistic_predictions = logistic_model.predict(X_test)

# Evaluation
linear_accuracy = accuracy_score(y_test, linear_predictions.round())
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print(f'Linear Regression Accuracy: {linear_accuracy}')
print(f'Logistic Regression Accuracy: {logistic_accuracy}')

# Additional metrics for logistic regression
logistic_classification_report = classification_report(y_test, logistic_predictions)
print('Logistic Regression Classification Report:\n', logistic_classification_report)
