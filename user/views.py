# cost_prediction/views.py
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import forms
from django.shortcuts import redirect
# Create your views here.
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import os
import pandas as pd
import numpy as np
from django.shortcuts import render
# from .forms import UserInputForm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import os


def encode_data(dataframe_series):
    if dataframe_series.dtype == 'object':
        return LabelEncoder().fit_transform(dataframe_series.astype(str))
    return dataframe_series


file_path = os.path.join(os.path.dirname(__file__), 'data.csv')

# Now, you can use this file_path in pd.read_csv()
df = pd.read_csv(file_path)

df = df.apply(encode_data)

data = df
# Preprocess the data if needed
# For demonstration, assume we drop any rows with missing values
data.dropna(inplace=True)

# Split data into features (X) and target (y)
X = data.drop('Construction Cost(BDT)', axis=1)
y = data['Construction Cost(BDT)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Train KNeighborsRegressor
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# Train XGBoostRegressor
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Define prediction function


def home(request):
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'Floors': [int(request.POST.get('floors'))],
            'Ordinary Rooms': [int(request.POST.get('rooms'))],
            'Building Height': [float(request.POST.get('height'))],
            'Floor Area': [float(request.POST.get('floor_area'))],
            'Brick Wall Area': [float(request.POST.get('brick_area'))],
            'Columns': [int(request.POST.get('cols'))],
            'Kitchens': [int(request.POST.get('kitchen'))],
            'Toilets': [int(request.POST.get('toilet'))],
            'Type of Foundation': [int(request.POST.get('foundation'))]
        }
        user_input_df = pd.DataFrame(user_input)

        # Make predictions using the trained models
        rf_predicted_cost = rf_model.predict(user_input_df)[0]
        knn_predicted_cost = knn_model.predict(user_input_df)[0]
        xgb_predicted_cost = xgb_model.predict(user_input_df)[0]

        # Pass prediction results to template
        return render(request, 'base.html', {'rf_predicted_cost': rf_predicted_cost,
                                             'knn_predicted_cost': knn_predicted_cost,
                                             'xgb_predicted_cost': xgb_predicted_cost,
                                             'mn': min(rf_predicted_cost, min(knn_predicted_cost, xgb_predicted_cost)),
                                             'mx': max(rf_predicted_cost, max(knn_predicted_cost, xgb_predicted_cost)),
                                             'avg': (rf_predicted_cost+knn_predicted_cost+xgb_predicted_cost)/3,
                                             })

    return render(request, 'base.html')
