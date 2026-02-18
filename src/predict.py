import pandas as pd
import joblib

# load model
model = joblib.load("credit_risk_model.pkl")

# load dataset structure
df = pd.read_csv("credit_data.csv")

# take one existing row and remove target column
new_customer = df.drop("credit_risk", axis=1).iloc[[0]].copy()

# modify values (simulate new person)
new_customer['age'] = 30
new_customer['amount'] = 3000
new_customer['duration'] = 24
new_customer['savings'] = '<=1000'
new_customer['empl oyment'] = '<=4'

# predict
prob_safe = model.predict_proba(new_customer)[0][1]

print("Repayment Probability:", prob_safe)

if prob_safe >= 0.7:
    print("Loan Approved")
else:
    print("Loan Rejected (High Risk)")
