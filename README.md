# Bank Marketing Optimization using Machine Learning
## 1. Business Problem
Banks invest heavily in outbound marketing campaigns (phone calls), but low conversion rates lead to inefficient resource allocation and high operational costs.

The key challenge is:
> How can the bank identify high-potential customers and improve campaign efficiency without increasing cost?

## 2. Objective
This project aims to:

- Predict customer likelihood to subscribe to a term deposit
- Optimize customer targeting strategy
- Improve conversion rate while reducing unnecessary outreach

## 3. Key Insights
- **Campaign fatigue effect**: Customers contacted more frequently are less likely to convert  
- **Customer history matters**: Previous successful interactions significantly increase conversion probability  
- **Strong segmentation opportunity**: Top predicted customers show significantly higher response rates (high lift)  
- **Macroeconomic factors** (e.g., interest rates) also influence customer decisions  

## 4. Modeling Approach
Three models were developed and compared:

- Logistic Regression (baseline, interpretable)
- Decision Tree (business-friendly rules)
- Random Forest (best predictive performance)

SMOTE was applied to address class imbalance.

## 5. Results
- Random Forest achieved the highest predictive performance (AUC)
- Logistic Regression provides strong interpretability for business insights
- Optimal classification threshold (~0.35) improves recall of high-value customers
- Top customer segments show significantly higher conversion rates (based on lift analysis)

## 6. Business Recommendation
Based on model results:

- **Prioritize top 30–40% high-probability customers** for marketing campaigns  
- **Reduce outreach to low-probability segments** to save cost  
- **Leverage customer history** to refine targeting strategy  
- **Optimize campaign frequency** to avoid over-contacting customers  

## 7. Business Impact (Estimated)
Applying the model-driven targeting strategy can:

- Reduce marketing costs by limiting low-value calls  
- Increase conversion efficiency by focusing on high-probability customers  
- Improve overall ROI of marketing campaigns  

## 8. Project Structure
- `data/` → dataset  
- `code/` → R scripts (data preprocessing, modeling, evaluation)  
- `report/` → project proposal & final report  
- `presentation/` → slides  

## 9. How to Run
1. Open R scripts in `code/`  
2. Load dataset from `data/`  
3. Run scripts sequentially  



## 10. Author

Nguyễn Hoàng Vân Nhi
