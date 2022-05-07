import matplotlib.pyplot as plt
import pandas as pd
appointment_data = pd.read_csv('../data/Medical_No_Shows.csv')
# initial data exploration
print(appointment_data.shape)
print(list(appointment_data.columns))
appointment_data.info()
appointment_data.isna().sum()  # no null values in dataset

# PatientID should be an int64 not an object
# TODO: take out hashtags and convert into int64
# TODO:
# Gender exploration
appointment_data['Gender'].value_counts()

'''since gender is only between M and F, 
one hot encoding could be used'''

%matplotlib inline
plt.imshow(appointment_data.corr(), cmap=plt.cm.GnBu,
           interpolation='nearest', data=True)
plt.colorbar()
tick_marks = [i for i in range(len(appointment_data.columns))]
plt.xticks(tick_marks, appointment_data.columns, rotation=45)
plt.yticks(tick_marks, appointment_data.columns, rotation=45)

'''
This is a supervised learning problem
since the existing data is already provided ahead of time

Model chosen is logistic regression
since we are trying to predict a binary output and
show the probability of each prediction of said binary output

Assumptions:
Binary logistic regression requires the dependent variable to be binary.
- this is true, either a patient was yes or no for no-show
For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome.
- Yes can be represented as 1, no as 0
Only the meaningful variables should be included.
- this will be determined after more analysis
The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
- this will be explored but from first glance all variables are independent
The independent variables are linearly related to the log odds.

Logistic regression requires quite large sample sizes.
- There are 110527 rows of data, which is sufficiently large
'''
