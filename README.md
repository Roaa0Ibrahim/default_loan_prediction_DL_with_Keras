# Lending Club Loan Data — Deep Learning with Keras

Capstone project for the **Deep Learning with TensorFlow & Keras** course, part of the Professional Certificate Program in AI & Machine Learning (Purdue University × IBM, delivered by Simplilearn). Completed in January 2025.

## Context
The project brief: use LendingClub's historical loan data (2007–2015) to build a deep learning model for lending decisions.

## Data
`loan_data.csv`: LendingClub borrowers with credit and loan features: interest rate, installment, debt-to-income ratio, FICO score, credit-line age, revolving utilisation, recent inquiries, delinquencies, public records, loan purpose, and repayment status.

## What I did
1. **Encoding:** one-hot encoded the loan `purpose`.
2. **Exploratory analysis and feature selection** with correlation analysis and box plots; kept 12 input features.
3. **Model:** a feed-forward neural network in Keras: 3 hidden layers of 32 units, LeakyReLU activations, He-uniform initialisation, Adam optimiser, binary cross-entropy; trained for 50 epochs with 15% validation split, on standardised inputs.
4. **Evaluated** with a confusion matrix, accuracy, a classification report, and training/validation accuracy and loss curves.

## Decisions I tested and rejected
- **SMOTE** to balance the classes: lowered accuracy, so not used. Dropping rows was not an option either, as the data is already small for deep learning.
- **Log transformation** for outliers: also lowered accuracy, so not used.

## Note on the target
In this notebook the model predicts **`credit.policy`** (whether a borrower meets LendingClub's underwriting criteria), and repayment status (`not.fully.paid`) is one of the input features.

## Tools
Python · TensorFlow · Keras · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Jupyter

## Files
- `default_loan_prediction_DL_with_Keras.ipynb` — analysis and model
- `loan_data.csv` — dataset
