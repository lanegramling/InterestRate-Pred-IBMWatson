#### Loan Interest Rate Predictor using IBM Watson

###### Dataset can be found [here](https://www.kaggle.com/wendykan/lending-club-loan-data "Lending Club Loan Data - Kaggle.com")

This is a basic interest rate predictor for loans, run on the provided dataset which contains ~2,260,000 rows x 145 columns. This was completed as a portion of the first project for AI class.

Two models are tested in this project:
- Multiple Linear Regression
- Ridge Regression

####Words of caution:

This is a rather mediocre application for a learning model, as many of the features used to feed the regressions for prediction exhibit high multi-collinearity due to their values being inextricably tied to the target variable (interest rate). The motivation for using Ridge Regression was an attempt to select a model better suited for this scenario, which includes an alpha parameter used to tone down the overfitting apparent in the multiple linear regression model. In using ridge regression, we chose to implement an iterative process to approximate the optimal the alpha value, which then results in the ridge regression showing a lower accuracy rating, which was the desired outcome.

A more useful approach would be to implement a similar model which instead predicts a loan's grade (A,AA,AAA,etc) based on the input features.

Last bit: This was implemented on IBM's Watson AI cloud, so there are some specifics here and there that would be unnecessary if co-opted for fiddling in a different environment.
