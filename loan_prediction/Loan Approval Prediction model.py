import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Loan (2).csv')

input_features = ['AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'Experience',
'LoanAmount', 'LoanDuration', 'MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus',
'MonthlyDebtPayments', 'CreditCardUtilizationRate', 'NumberOfOpenCreditLines',
'NumberOfCreditInquiries', 'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose',
'PreviousLoanDefaults', 'PaymentHistory', 'NetWorth', 'MonthlyLoanPayment',
'TotalDebtToIncomeRatio', 'TotalLiabilities', 'LoanApproved']

LoanDF = data[input_features]

categories = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']

le = LabelEncoder()
LoanDF['EmploymentStatus'] = le.fit_transform(LoanDF['EmploymentStatus'])
LoanDF['EducationLevel'] = le.fit_transform(LoanDF['EducationLevel'])
LoanDF['MaritalStatus'] = le.fit_transform(LoanDF['MaritalStatus'])
LoanDF['HomeOwnershipStatus'] = le.fit_transform(LoanDF['HomeOwnershipStatus'])
LoanDF['LoanPurpose'] = le.fit_transform(LoanDF['LoanPurpose'])

X = LoanDF.drop(columns=['LoanApproved'])
y = LoanDF['LoanApproved']

num_cols = X.select_dtypes(include=[np.number]).columns
scaler = preprocessing.StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)


print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))




actual = y_test.value_counts()
print("Actual Loan Approval counts:")
print(actual)
predicted = pd.Series(y_pred).value_counts()
print("\nPredicted Loan Approval counts:")
print(predicted)

percentage = actual / predicted
print("The proportion of actual(Accepted & Rejected) to predicted (Accepted & Rejected) loans is: ", round(percentage, 2))
