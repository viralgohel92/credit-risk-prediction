# Credit Risk Prediction System

## Problem

Banks lose money when customers default on loans.
The goal of this project is to predict whether a loan applicant is risky before approval.

## Solution

Built a machine learning credit scoring pipeline using Logistic Regression with class imbalance handling and decision threshold tuning.

## Key Features

* Full ML pipeline (encoding + scaling + model)
* Handles categorical financial data
* Probability-based risk scoring
* Custom approval threshold (risk policy control)
* Saved model for real-time prediction

## Model Performance

* ROC-AUC: 0.76
* Defaulter Recall improved from 45% → 70%
* Threshold tuning captured 93% of risky borrowers

## Business Insight

Instead of maximizing accuracy, the system prioritizes minimizing financial loss by detecting defaulters.

## How to Run

```bash
pip install -r requirements.txt
python src/predict.py
```
