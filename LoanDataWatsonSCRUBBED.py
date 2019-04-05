#LIBRARIES
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3
import re
import statsmodels.api as sm
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

#CONSTANTS
# For use with conversion of grades to numeric
# A:0 B:5 C:10 D:15 E:20 F:25 G:30
GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# For use with conversion of home ownership to numeric
# RENT:0  MORTGAGE:1  OWN:2  OTHER:3  NONE:4  ANY:5
HOME_OWNERSHIP = ['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE', 'ANY']

pd.set_option('mode.chained_assignment', None)

#############################################
################ SCRUBBED ###################
IBM_API_KEY = '********SCRUBBED********'
#############################################

#READ IN DATA
def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_72476f18d63244a8bbe76c7fc0938d89 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=IBM_API_KEY',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_72476f18d63244a8bbe76c7fc0938d89.get_object(Bucket='groupproject1-donotdelete-pr-h9ebopxflvw5ly',Key='actually_scrubbed.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_4 = pd.read_csv(body)


#ATTACH DATA AND STANDARDIZE DATA
df_predictors = df_data_4[['loan_amnt','funded_amnt','dti','annual_inc','open_acc','revol_util','total_acc','term','grade','sub_grade','home_ownership','emp_length']]

# Scrub occasional 'n/a' values to -1
df_predictors.replace('n/a', -1)
df_predictors.replace(np.NaN, -1)
df_predictors.replace('nan',-1)
# Convert 'term' column to numeric values
df_predictors['term'] = df_predictors['term'].apply(lambda x: int(re.sub(r"\D", "", x)))

# Convert 'grade' and 'sub_grade' columns to numeric weights
df_predictors['grade'] = df_predictors['grade'].apply(lambda x: GRADES.index(x) * 5)
df_predictors['sub_grade'] = df_predictors['sub_grade'].apply(lambda x: (GRADES.index(x[0]) * 5) + int(x[1]))

# Convert home ownership values to numeric weights
df_predictors['home_ownership'] = df_predictors['home_ownership'].apply(lambda x: HOME_OWNERSHIP.index(x))

# Convert employee length columns into numeric values.
df_predictors['emp_length'] = df_predictors['emp_length'].apply(lambda x: int(re.sub(r"\D", "", str(x)) or -1))

#Values we are trying to predict
target = df_data_4['int_rate']



#LINEAR REGGRESSION
model = sm.OLS(target.values, df_predictors.values).fit()
predictions = model.predict(df_predictors.values)

print(model.summary())
print("")
print("Predictions for Interest Rate")
print(predictions)
print("Sum of Squares Error")
print(rss)
print("Mean Squared Error for Linear Regression Interest Rate")
print(mse)
print("")

#Error for Linear Regression
rss = sum((predictions-target)**2)
#mse = mean_squared_error(target, predictions)
mse = (sum((predictions-target)**2))/886727

scaler = StandardScaler()
X_std = scaler.fit_transform(df_predictors)

regr_cv = RidgeCV(alphas=[5.0, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
model_cv = regr_cv.fit(X_std, target)
print("Alpha")
print(model_cv.alpha_)
print("")


#RIDGE REGRESSION
def ridge_regression():
    #Fit the model
    ridgereg = Ridge(alpha=model_cv.alpha_,normalize=True)
    ridgereg.fit(X_std,target)
    y_pred = ridgereg.predict(X_std)
    score1 = ridgereg.score(X_std,target)
    print("Predictions for Ridge Regression")
    print(y_pred)
    print("Intercept")
    print(ridgereg.intercept_)
    print("Coeffcients")
    print(ridgereg.coef_)
    print("Score")
    print(score1)
    
    #Error Ridge Regression
    print("Sum of Squares Error")
    rss = sum((y_pred-target)**2)
    print(rss)
    print("Mean Squared Error")
    mse = mean_squared_error(target, y_pred)
    print(mse)
    
print("Ridge Regression: \n")
ridge_regression()
print("")


#Correlation Matrix
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)

print("Correlation Matrix")
plot_corr(df_predictors.join(df_data_4[['int_rate']]))


/* Output
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.991
Model:                            OLS   Adj. R-squared:                  0.991
Method:                 Least Squares   F-statistic:                 8.016e+06
Date:                Tue, 19 Mar 2019   Prob (F-statistic):               0.00
Time:                        03:29:36   Log-Likelihood:            -1.5133e+06
No. Observations:              886727   AIC:                         3.027e+06
Df Residuals:                  886715   BIC:                         3.027e+06
Df Model:                          12                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0002   4.38e-06    -43.038      0.000      -0.000      -0.000
x2             0.0002   4.38e-06     38.000      0.000       0.000       0.000
x3             0.0028   8.42e-05     33.254      0.000       0.003       0.003
x4          1.297e-06   2.36e-08     54.914      0.000    1.25e-06    1.34e-06
x5             0.0444      0.000    120.554      0.000       0.044       0.045
x6             0.0207   5.81e-05    355.565      0.000       0.021       0.021
x7             0.0139      0.000     82.107      0.000       0.014       0.014
x8             0.0551      0.000    424.711      0.000       0.055       0.055
x9            -0.2456      0.001   -261.438      0.000      -0.247      -0.244
x10            0.8953      0.001    926.166      0.000       0.893       0.897
x11            0.1352      0.002     59.839      0.000       0.131       0.140
x12            0.0482      0.000    127.848      0.000       0.047       0.049
==============================================================================
Omnibus:                    65908.050   Durbin-Watson:                   1.288
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           215194.098
Skew:                          -0.361   Prob(JB):                         0.00
Kurtosis:                       5.303   Cond. No.                     1.61e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.61e+05. This might indicate that there are
strong multicollinearity or other numerical problems.

Predictions for Interest Rate
[  9.49201066  13.80414767  15.69050882 ...,  16.9030273   20.59711118
  12.05478141]
Sum of Squares Error
1576454.266571791
Mean Squared Error for Linear Regression Interest Rate
1.7778349667618003

Alpha
6.0

Ridge Regression: 

Predictions for Ridge Regression
[ 12.56776133  13.46592761  13.61061245 ...,  14.51056788  15.16796186
  12.87746217]
Intercept
13.2457729042
Coeffcients
[ 0.0505478   0.05071321  0.03428691 -0.04169603 -0.00638414  0.12679751
 -0.02133495  0.19566363  0.50453848  0.52075168 -0.02505945  0.00057838]
Score
0.442230517934
Sum of Squares Error
9495468.594870036
Mean Squared Error
10.7084464495

Correlation Matrix

(img)
*/