# Employee Attrition Analysis and Prediction

**Live app:** https://hr-attrition-prediction-justtry.streamlit.app

## Problem
Predict which employees are likely to leave a company, and find the factors behind attrition.

## Dataset
HR dataset with 311 employees (104 left, 33%). Target: `Termd` (1 = left).

## Methods
Data cleaning, exploratory analysis, K-Means segmentation, and four classifiers (Logistic Regression, Decision Tree, Random Forest, SVM) compared with 5-fold cross-validation. Names, IDs, dates and personal attributes (gender, race, marital status) were excluded to avoid leakage and bias.

## Results
| Model | Accuracy | Recall | F1 | CV F1 |
|---|---|---|---|---|
| SVM (selected) | 0.698 | 0.667 | 0.596 | 0.572 |
| Logistic Regression | 0.635 | 0.667 | 0.549 | 0.568 |
| Random Forest | 0.698 | 0.619 | 0.578 | 0.528 |
| Decision Tree | 0.635 | 0.381 | 0.410 | 0.399 |

## Key Insights
- Production has the most leavers (about 40% attrition).
- Google Search hires leave at 61%; Employee Referral hires at 16%.
- Three managers account for about 37% of leavers.
- Salary, engagement and absences are the top model features, but each is a weak predictor alone.

## Files
- `HR_Attrition_Analysis.ipynb`: full analysis
- `app.py`: Streamlit app
- `model.pkl`, `options.pkl`: trained model and dropdown options
- `hr_cleaned_data.csv`: dataset

## Tools
Python, Pandas, Scikit-learn, Seaborn, Matplotlib, Streamlit

## Limitations
Small dataset; the model is a screening aid, not a decision-maker.
