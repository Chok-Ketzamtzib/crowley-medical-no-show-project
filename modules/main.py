import pandas as pd
appointment_data = pd.read_csv('../data/Medical_No_Shows.csv')
# initial data exploration
appointment_data.info()
appointment_data.isna().sum()

# PatientID should be an int64 not an object
# TODO: take out hashtags and convert into int64 

# Gender exploration
appointment_data['Gender'].value_counts()

'''since gender is only between M and F, 
one hot encoding could be used'''

