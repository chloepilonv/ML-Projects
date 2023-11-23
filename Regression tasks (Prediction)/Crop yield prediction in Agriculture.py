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
import seaborn as sns
import matplotlib.pyplot as plt

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

#FIRST CLEANING 
#SHow the variance of numerical values
trainnum =train.select_dtypes(include=np.number) 
trainnum.var()
#We could drop Acre, NoFertilizerAppln because <0.1
train=train.drop(['Acre','NoFertilizerAppln'],axis=1)
#Drop duplicate rows
train=train.drop_duplicates()

#FIRST FEATURE SELECTION (NUMERICAL)
X=train.drop(['ID'],axis=1)
numerical_X = X.select_dtypes(include=np.number) 

# Calculating the correlation matrix
corr_matrix_numerical_X = numerical_X.corr()

# Plotting the heatmap with annotations
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_numerical_X, annot=True, cmap='coolwarm', fmt=".2f") #with annot=True, each cell will be annotated with the numeric value
plt.title("Correlation Heatmap (Numerical Variables)")
plt.show()

#We will keep only features that have a correlation of 0.30 or more with the target variable.
#Also, we see that Harv_hand_rent is highly correlated with 2tdUrea - which is strongly correlated with Yield -, so we will keep it
#With this same reasoning, we could do the same with CultLand and CropCultLand in regard to BasalDAP, BasalUrea and 2tdUrea, but we won't for now because they have lower values
yield_correlations = corr_matrix_numerical_X['Yield']
# Keeping only highly correlated features over > 0.3 AND Harv_hand_rent
selected_numfeatures = yield_correlations[abs(yield_correlations) > 0.30].index.tolist()

if 'Harv_hand_rent' not in selected_numfeatures:
    selected_numfeatures.append('Harv_hand_rent')

# Now selected_columns holds the names of columns with correlation > 0.30 with 'Yield'
# and includes 'Harv_hand_rent'
print(selected_numfeatures)


#ENCODING CATEGORICAL DATAS

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
