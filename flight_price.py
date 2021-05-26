### importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from feature_engineering import convert_journey,convert_departure,convert_arrival,time_taken,replace_stops,\
    to_categorical

from visualization_analysis import highest_price,source_high_price,count_flight_monthwise,count_airlines,\
    corelation

from model import train_test,feature_importance,random_forest,random_search_CV

sns.set()

### reading the dataset

flight = pd.read_excel(r'D:\DataScienceDeloymentProjects\Flight Fare prediction\Data_Train.xlsx',engine='openpyxl')

print(flight.head)
print(flight.shape)

### checking for null values
print('Finding the null values in every column if exists',flight.isnull().sum())


### FEATURE ENGINEERING process from feature engineering.py file

## 1. drop null values
flight.dropna(inplace = True)

### 2. convert date of journey to datetime
data = convert_journey(flight)

### 3. convert departure time to hours and minutes seperately
data1 = convert_departure(data)

### 4. convert arrival time to hours and minnutes column seperately
data2 = convert_arrival(data1)

#### 5. convert duration time to hours and minutes
data3 = time_taken(data2)
data3.drop(['Route','Additional_Info'],axis=1,inplace=True)

### 6. replace stops to numeric ordinal value using dictionary
data4 = replace_stops(data3)

#### 7. one hot encoding and label encoding of categorical data and concatenating it into final dataframe

data_train = to_categorical(data4)
data_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)


#### visualization functions to display the plots for analysis from visualization_analysis.py file
highest_price(data3)
source_high_price(data3)
count_flight_monthwise(data3)
count_airlines(data3)





########################### Test Data ######################################

test = pd.read_excel(r'D:\DataScienceDeloymentProjects\Flight Fare prediction\Test_set.xlsx',engine='openpyxl')

###  FEATURE ENGINEERING ON TEST DATA similar to traning data

## 1. drop null values
test.dropna(inplace = True)

### 2. convert date of journey to datetime
test_1 = convert_journey(test)

### 3. convert departure time to hours and minutes seperately
test_2 = convert_departure(test_1)


### 4. convert arrival time to hours and minnutes column seperately
test_3 = convert_arrival(test_2)


#### 5. convert duration time to hours and minutes
test_4 = time_taken(test_3)
test_4.drop(['Route','Additional_Info'],axis=1,inplace=True)


### 6. replace stops to numeric ordinal value using dictionary
test_5 = replace_stops(test_4)


data_test = to_categorical(test_5)
data_test.drop(['Airline','Source','Destination'],axis=1,inplace=True)


####### model training and prediction ###############
X,y = train_test(data_train)
###train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



### RANDOM FORESTS ALGORITHM
train_score,test_score = random_forest(X_train,X_test,y_train,y_test)
print("The training score for random forests is " , train_score)
print("The test score using random forests is ", test_score)

#### RANDOM SEARCH CV 
prediction = random_search_CV(X_train,X_test,y_train,y_test)


### dumping the model into a file using pickle




### feature selection plots
corelation(flight)
feature_importance(X,y)








