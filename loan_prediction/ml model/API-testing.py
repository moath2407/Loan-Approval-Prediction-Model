import requests

# Check if API root is responding
#r = requests.get("http://127.0.0.1:8000/")
#print(r.status_code)
#print(r.json())

# Send POST request with loan application JSON


r_apply = requests.post(
        "http://127.0.0.1:8000/predict/apply",
        json=
{
    "AnnualIncome": 52825,
    "CreditScore": 572,
    "EmploymentStatus": "Employed",
    "EducationLevel": "Bachelor",
    "Experience": 17,
    "LoanAmount": 14135,
    "LoanDuration": 24,
    "MaritalStatus": "Single",
    "NumberOfDependents": 0,
    "HomeOwnershipStatus": "Own",
    "MonthlyDebtPayments": 549,
    "CreditCardUtilizationRate": 0.17613715546323208,
    "NumberOfOpenCreditLines": 2,
    "NumberOfCreditInquiries": 1,
    "DebtToIncomeRatio": 0.2548345044296835,
    "BankruptcyHistory": 0,
    "LoanPurpose": "Education",
    "PreviousLoanDefaults": 0,
    "PaymentHistory": 26,
    "NetWorth": 52931,
    "MonthlyLoanPayment": 731.5003940899489,
    "TotalDebtToIncomeRatio": 0.2908850871572056,
    "TotalLiabilities": 33289
    
}
    )


print(r_apply.json())
