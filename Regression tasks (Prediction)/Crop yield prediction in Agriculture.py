# Import libraries
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import xgboost as xgb

# Load files. Current datas are private and can't be share on Github. Will correlate to another public data sets.
data_path = ''
train = pd.read_csv(data_path + '') #
test = pd.read_csv(data_path + '')
sample_submission = pd.read_csv(data_path + '')
var_desc = pd.read_csv(data_path + '')

# Preview files
train.head()
test.head()
sample_submission.head()
var_desc.head()

# ENCODE DATAS

#New data set without column (axis=1) 'ID' (not predictive of the outcome) and 'Yield' (our target variable) from X
X = train.drop(['ID', 'Yield'], axis = 1) 
#Keeps only the columns with numerical values. NDLR : But...
#X =X.select_dtypes(include=np.number) 

#The dtype of the  YYYY-MM-DD in the column 'CropTillageDate' don't even need to be change. 
#Label Encoding for dates. Turns the dates into numerical values in ascending order. 

labelencoder = LabelEncoder()
#Transforms only the columns associated to CropTillageDate, RcNursEstDate, SeedingSowingTransplanting, Harv_date and CropHarvestDate in X
X['CropTillageDate']= labelencoder.fit_transform(X['CropTillageDate'])
X['RcNursEstDate']= labelencoder.fit_transform(X['RcNursEstDate'])
X['SeedingSowingTransplanting']= labelencoder.fit_transform(X['SeedingSowingTransplanting'])
X['Harv_date']= labelencoder.fit_transform(X['Harv_date'])
X['Threshing_date']= labelencoder.fit_transform(X['Threshing_date'])
print(X['Threshing_date'].head())

#One hot encoding for categorical values
onehotencoder = OneHotEncoder(sparse_output=False)
#Transforms the categorical values into numerical values for all the columns with categorical values automatically
X = onehotencoder.fit_transform(X)

#Special case : Multiple choices ('LandPreparationMethod', 'NurseDetFactor' and 'TransDetFactor')

#Speciale case 2 : Subcategories. For example : Manuel X, Manuel Y, Manuel Z and Broadcasting. 

#sets the target variable
y = train.Yield 


#FEATURE SELECTION



#TRAINING
 #splittings the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1234)

#MODEL
#This section provides different alternatives to consider NAN VALUES, keeping the one with highest RMSE and Acc. 
# ORIGINAL RANDOM FOREST CODE
model = RandomForestRegressor(random_state = 1234) #Ensemble learning method based on decision trees
#Other option : GradientBoostingRegressor

#ORIGINAL MODEL FIT (REPLACE NAN WITH 0)
model.fit(X_train.fillna(0), y_train)

#ORIGINAL MODEL FIT + REMOVE NAN VALUES
#X_train = X_train.dropna()
#y_train = y_train.dropna()
#model.fit(X_train, y_train)

# ORIGINAL MODEL FIT + REPLACE NAN WITH MEAN
#model.fit(X_train.fillna(X_train.mean()), y_train)

# ORIGINAL MODEL FIT + REPLACE NAN WITH MEDIAN (BETTER THAN MEAN)
#model.fit(X_train.fillna(X_train.median()), y_train)

#ORIGINAL MODEL FIT + REPLACE NAN WITH LINEAR REGRESSION

#NEW MODEL : EXTREME GRADIENT BOOSTING
#model = xgb.XGBRegressor(random_state = 1234)
#FIT THE EXTREME GRADIENT BOOSTING MODEL THAT CAN HANDLE NAN VALUES
#model.fit(X_train, y_train)

#PREDICTIONS
# Make predictions
preds = model.predict(X_test.fillna(0)) #predicting the target variable for the test set

#MEASURE MODEL PERFORMANCE
rmse = mean_squared_error(y_test, preds, squared=False) #calculating the RMSE. squared=False means that we want the RMSE and not the MSE
acc = model.score(X_test.fillna(0), y_test) #calculating the accuracy of the model
acc2 = accuracy_score(y_test, preds.round(), normalize=False) #calculating the accuracy of the model
acctotal = acc2/len(y_test)
mae = np.mean(abs(preds - y_test))
r2 = r2_score(y_test, preds) #calculate r square

print('RMSE:', rmse)
print('ACC:', acctotal)
print('R2:', r2)
print('MAE:', mae)

# Output predictions
test_df = test[X.columns] #selecting the same columns as in the training set. The features are the same, but the values are different
preds = model.predict(test_df.fillna(0)) #predicting the target variable for the test set

# Create file
sub = pd.DataFrame({'ID': test.ID, 'Yield': preds})
sub.to_csv('Results.csv', index = False)

sub.head()