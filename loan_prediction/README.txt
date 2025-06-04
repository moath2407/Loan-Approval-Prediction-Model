Loan Approval Prediction Model
This project uses a logistic regression model to predict whether a loan application will be approved or rejected, based on applicant financial and demographic data.

------------------------------------------------------------------------------
## Overview
The dataset includes features such as income, credit score, employment status, loan amount, and more. The goal is to build a machine learning model that can classify loan applications as either approved (1) or not approved (0) using these attributes.

------------------------------------------------------------------------------

## Methods Used
The process begins by selecting relevant features from the dataset. Categorical variables like Employment Status, Education Level, and Loan Purpose are encoded numerically using Label Encoding to make them compatible with machine learning algorithms. Numerical features are standardized using Z-score normalization with StandardScaler to ensure all values are on a comparable scale.

The data is split into training and testing sets (80/20 split). A Logistic Regression classifier is trained on the processed data, which is well-suited for binary classification tasks. The model is then evaluated using accuracy score and a classification report (precision, recall, f1-score).

Finally, the model's predicted approval results are compared with the actual loan approval data, and the proportions of prediction accuracy are printed for interpretability.

------------------------------------------------------------------------------

## Output
The following outputs are generated:

Accuracy Score: Shows how well the model performs overall.

Classification Report: Detailed breakdown of precision, recall, and F1-score.

Actual vs Predicted Counts: Shows the distribution of approvals/rejections.

Proportional Comparison: Compares actual outcomes to model predictions.

------------------------------------------------------------------------------

## Requirements
Python 3
pandas
numpy
matplotlib
seaborn
scikit-learn

------------------------------------------------------------------------------

## How to Run
Place the dataset file named Loan.csv in your project directory.

Run the Python script.

Observe the printed model evaluation and loan approval predictions.