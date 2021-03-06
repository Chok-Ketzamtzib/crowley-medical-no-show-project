from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
plt.rc("font", size=14)
appointment_data = pd.read_csv('../data/Medical_No_Shows.csv')
# initial data exploration
print(appointment_data.shape)
print(list(appointment_data.columns))
appointment_data.info()
appointment_data.isna().sum()  # no null values in dataset

print(len(appointment_data['PatientID'].unique()))
# PatientID should be an int64 not an object
# TODO: take out hashtags and convert into int64
# TODO:
# Gender exploration
appointment_data['Gender'].value_counts()
appointment_data['Disability'].value_counts()
#  Disability has values outside of what is defined
# TODO: cleanup Disability column

appointment_data[appointment_data['Disability'] == 2] = 1
appointment_data[appointment_data['Disability'] == 3] = 1
appointment_data[appointment_data['Disability'] == 4] = 1

'''since gender is only between M and F, 
one hot encoding could be used'''

appointment_data['No-show'].value_counts()
sns.countplot(x='No-show', data=appointment_data, palette='hls')
plt.show()
plt.savefig('count plot')

count_no_show = len(appointment_data[appointment_data['No-show'] == 'No'])
count_show = len(appointment_data[appointment_data['No-show'] == 'Yes'])
pct_of_no_sub = count_no_show/(count_no_show+count_show)
print("percentage of no-show is", pct_of_no_sub*100)
pct_of_sub = count_show/(count_no_show+count_show)
print("percentage of show", pct_of_sub*100)
# classes are imbalanced, 80:20, rebalancing needed

appointment_data.groupby('No-show').mean()

'''
Obervations
average age of patients who do not show up is lower than patients who did show up
More medicaid patients on average do not show up than non-medicaid patients
Patients with hypertension on average show up less than patients with hypertension
Patients receiving SMS on average show up less than patients with no SMS reminder

'''
#  No-show Frequency Gender
%matplotlib inline
pd.crosstab(appointment_data['Gender'],
            appointment_data['No-show']).plot(kind='bar')
plt.title('No-show Frequency for Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency of No-shows')
plt.savefig('No_show_frequency_gender')

#  No-show Frequency Hypertension
%matplotlib inline
pd.crosstab(appointment_data['Hypertension'],
            appointment_data['No-show']).plot(kind='bar')
plt.title('No-show Frequency for Hypertension')
plt.xlabel('Hypertension')
plt.ylabel('Frequency of No-shows')
plt.savefig('No_show_frequency_hypertension')

# No-show Frequency Diabetes
%matplotlib inline
pd.crosstab(appointment_data['Diabetes'],
            appointment_data['No-show']).plot(kind='bar')
plt.title('No-show Frequency for Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Frequency of No-shows')
plt.savefig('No_show_frequency_diabetes')

# No-show Frequency Alcoholism
%matplotlib inline
pd.crosstab(appointment_data['Alcoholism'],
            appointment_data['No-show']).plot(kind='bar')
plt.title('No-show Frequency for Alcoholism')
plt.xlabel('Alcoholism')
plt.ylabel('Frequency of No-shows')
plt.savefig('No_show_frequency_alcoholism')
# No-show Frequency Disability
%matplotlib inline
pd.crosstab(appointment_data['Disability'],
            appointment_data['No-show']).plot(kind='bar')
plt.title('No-show Frequency for Disability')
plt.xlabel('Disability')
plt.ylabel('Frequency of No-shows')
plt.savefig('No_show_frequency_disability')
# No-show Frequency SMS_Received
%matplotlib inline
pd.crosstab(appointment_data['SMS_received'],
            appointment_data['No-show']).plot(kind='bar')
plt.title('No-show Frequency for SMS_received')
plt.xlabel('SMS_received')
plt.ylabel('Frequency of No-shows')
plt.savefig('No_show_frequency_SMS')

# Stacked bar charts to compare classes with each other without the frequency
table = pd.crosstab(appointment_data['Gender'], appointment_data['No-show'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Gender vs No-show')
plt.xlabel('Gender')
plt.ylabel('Proportion of No-shows')
plt.savefig('gender_vs_noshow_stack')
# Despite a higher number of women keeping appointments than men,
# really at the same rate both genders are no-shows
table = pd.crosstab(
    appointment_data['Hypertension'], appointment_data['No-show'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Hypertension vs No-show')
plt.xlabel('Hypertension')
plt.ylabel('Proportion of No-shows')
plt.savefig('hypertension_vs_noshow_stack')
# hypertension seems like a good predictor
table = pd.crosstab(appointment_data['Diabetes'], appointment_data['No-show'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Diabetes vs No-show')
plt.xlabel('Diabetes')
plt.ylabel('Proportion of No-shows')
plt.savefig('diabetes_vs_noshow_stack')
# diabetes does not seems like a good predictor
table = pd.crosstab(
    appointment_data['Alcoholism'], appointment_data['No-show'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Alcoholism vs No-show')
plt.xlabel('Alcoholism')
plt.ylabel('Proportion of No-shows')
plt.savefig('alcoholism_vs_noshow_stack')
# Alcoholism seems to have the same rate
table = pd.crosstab(
    appointment_data['Disability'], appointment_data['No-show'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Disability vs No-show')
plt.xlabel('Disability')
plt.ylabel('Proportion of No-shows')
plt.savefig('disability_vs_noshow_stack')
# Disability seems like a good predictor
table = pd.crosstab(
    appointment_data['SMS_received'], appointment_data['No-show'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of SMS_received vs No-show')
plt.xlabel('SMS_received')
plt.ylabel('Proportion of No-shows')
plt.savefig('SMS_vs_noshow_stack')
# SMS seems like a good predictor


pd.crosstab(appointment_data['AppointmentDay'],
            appointment_data['No-show']).plot(kind='bar')
plt.title('No-show Frequency for Appointment Day')
plt.xlabel('Appointment Day')
plt.ylabel('Frequency of No-shows')
plt.savefig('apptday_fre_noshow_bar')

appointment_data['Age'].hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')
# most people are in the range of 0-60


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

# Features to focus on for now for LR model
'''
Gender            0
Age               0
LocationID        0
MedicaidIND       0
Hypertension      0
Diabetes          0
Alcoholism        0
Disability        0
SMS_received      0
'''

# One Hot Encode Gender
hot_list = pd.get_dummies(appointment_data['Gender'], prefix='hot')
new_data = appointment_data.join(hot_list)
appointment_data = new_data

# data_vars = appointment_data.columns.values.tolist()
# to_keep = [i for i in data_vars if i not in one_hot_vars]
# data_final=appointment_data[to_keep]
data_final = appointment_data
data_final.columns.values

data_final.drop(['PatientID', 'AppointmentID',
                'ScheduledDay', 'AppointmentDay', 'Gender'], axis=1, inplace=True)
data_final['No-show'].replace(['No', 'Yes'], [0, 1], inplace=True)

data_final = data_final.astype('uint8')
data_final.columns.values
data_final.info()
'''
Model Setup

'''
X = data_final.loc[:, data_final.columns != 'No-show']
y = data_final.loc[:, data_final.columns == 'No-show']


os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X, os_data_y = os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['No-show'])
# we can Check the numbers of our data
print("length of oversampled data is ", len(os_data_X))
print("Number of not no-shows in oversampled data",
      len(os_data_y[os_data_y['No-show'] == 0]))
print("Number of no-shows", len(os_data_y[os_data_y['No-show'] == 1]))
print("Proportion of not no-shows data in oversampled data is ",
      len(os_data_y[os_data_y['No-show'] == 0])/len(os_data_X))
print("Proportion of no-shows data in oversampled data is ",
      len(os_data_y[os_data_y['No-show'] == 1])/len(os_data_X))

# Recursive Feature Elimination
data_final_vars = data_final.columns.values.tolist()
y = ['No-show']
X = [i for i in data_final_vars if i not in y]
logreg = LogisticRegression(max_iter=10000)
rfe = RFE(logreg, n_features_to_select=11)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
# All features pass as true, so all features are significant
# for the model

# Putting X and y to original data

X = os_data_X
y = os_data_y.values.ravel()

# Print out Feature Importance
logreg.fit(X, y)
importance = logreg.coef_[0]
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
plt.bar([x for x in range(len(importance))], importance)
plt.title(
    'Bar chart of Logistic Regression Coefficients as Feature Importance Scores')
plt.xlabel('Feature[0-10]')
plt.ylabel('Feature Importance Score')
plt.show()
#  Model Implementation
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())

# Model fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#  Predicting test results
y_pred = logreg.predict(X_test)

# Printing the Probabilities and showcasing in dataframe
y_probs = logreg.predict_proba(X_test)

# Calculate Accuracy
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
    logreg.score(X_test, y_test)))


#  Show graph of logistic regression for better understanding
# plt.scatter(X,logreg.predict_proba(X)[:,1])
sns.regplot(x='SMS_received', y='No-show', data=data_final, logistic=True)
# Validation with Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
# Confusion matrix does not look right TODO: investigate further

print(classification_report(y_test, y_pred))

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
