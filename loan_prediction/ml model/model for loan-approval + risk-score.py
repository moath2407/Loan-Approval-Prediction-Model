import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle





data = pd.read_csv('Loan.csv')
data = data.drop(columns=['ApplicationDate'])  

#print(data.head())
#print(data.describe())
#print(data.info())


#Heatmap of correlation
#data_numerical_columns = data.select_dtypes(include = ['number'])
#plt.figure(figsize = (50,50))
#sns.heatmap(data_numerical_columns.corr(), annot = True)
#plt.show()


input_features = ['AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'Experience',
'LoanAmount', 'LoanDuration', 'MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus',
'MonthlyDebtPayments', 'CreditCardUtilizationRate', 'NumberOfOpenCreditLines',
'NumberOfCreditInquiries', 'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose',
'PreviousLoanDefaults', 'PaymentHistory', 'NetWorth', 'MonthlyLoanPayment',
'TotalDebtToIncomeRatio', 'TotalLiabilities']

LoanDF = data[input_features]
#print(LoanDF.describe)


#Handling categorical variables and encoding them
##only encode categorical variables (variables that are based on categories (male, married etc.))
###Ordinal encoding converts categories into binary/trueorfalse variables

categorical_var = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
encoded_var = pd.get_dummies(LoanDF, columns=categorical_var, drop_first=True)
#print(encoded_var)

#Standardize using standardscaler()
scaler = StandardScaler()
#Seperate the numerical columns into a seperate list
numerical_cols= [
    'AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration',
    'NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines',
    'NumberOfCreditInquiries','DebtToIncomeRatio','NetWorth',
    'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'TotalLiabilities' ]

#The encoded_var now has standardized numerical columns, and encoded categorical columns
encoded_var[numerical_cols] = scaler.fit_transform(encoded_var[numerical_cols])
#print(encoded_var)

#Training the model
X = encoded_var[:] #All features
y_risk = data['RiskScore']
y_loanapproval = data['LoanApproved']

#You should split the data into a risk set and approval set
X_trainrisk, X_testrisk, y_trainrisk, y_testrisk = train_test_split(X, y_risk, test_size = 0.2, random_state = 18)
X_trainapproval, X_testapproval, y_trainapproval, y_testapproval = train_test_split(X, y_loanapproval, test_size = 0.2, random_state = 18)

#Creating a linear model
regressor = LinearRegression()
regressor.fit(X_trainrisk, y_trainrisk)
y_predictrisk = regressor.predict(X_testrisk)

#Evaluate using MSE and R2
print("Mean Squared Error: ", round(mean_squared_error(y_testrisk, y_predictrisk), 2))
print("R2 score: ", round(r2_score(y_testrisk, y_predictrisk), 2))
#Scatter plot
x_axis=range(len(y_testrisk))
plt.scatter(x_axis, y_testrisk, label='Actual Risk Score', color='red')
plt.scatter(x_axis, y_predictrisk, label='Predicted Risk Score', color='blue')
plt.xlabel('Test Data Points')
plt.ylabel('Risk Score')
plt.legend()
plt.show()

#Classification - Gaussian is the simplest to use
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_trainapproval, y_trainapproval)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_testapproval)
print("Accuracy using Gaussian Classification:", accuracy_score(y_testapproval, y_pred))


#Another Classification method - TO COMPARE ACCURACIES - RFC
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_trainapproval, y_trainapproval)
y_pred_approval = classifier.predict(X_testapproval)
print("Accuracy using RFC:", accuracy_score(y_testapproval, y_pred_approval))

#Another Classification method - TO COMPARE ACCURACIES - KNearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_trainapproval, y_trainapproval)
y_pred_2 = clf.predict(X_testapproval)
print("Accuracy using KNN: ", accuracy_score(y_testapproval, y_pred_2))

#Another Classification method - TO COMPARE ACCURACIES - LinearSVC
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_trainapproval, y_trainapproval)
y_pred_3 = clf.predict(X_testapproval)
print("Accuracy usinng LinearSVC: ", accuracy_score(y_testapproval, y_pred_3))


#How to pickle:
with open ('model_regressor.pkl', 'wb') as f:
    pickle.dump(regressor, f)
with open('model_Linearclassifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('model_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model_features.pkl', 'wb') as f:
    pickle.dump(list(encoded_var.columns), f)

