# Bank Subscription Propensity Model

# Bank Marketing: Subscription Propensity Pipeline

**Predictive modeling (Random Forest, XGBoost, KNN) to optimize term-deposit conversion and marketing ROI.**

**End-to-end ML pipeline to predict customer propensity for term-deposit subscriptions using XGBoost, SHAP, and SMOTE.**

**Executive Summary**

In the highly competitive retail banking sector, identifying high-potential customers is more cost-effective than broad-spectrum advertising. This project leverages Predictive Analytics to determine the likelihood of a client subscribing to a term deposit. By moving from "blind" calling to "data-driven" targeting, the organization can proactively allocate resources to high-propensity leads, significantly enhancing conversion rates and marketing efficiency.

**Problem Statement**

Traditional marketing efforts in the organization lack precision, often relying on general demographics without deep insights into behavioral or economic drivers. This leads to:

- Operational Inefficiency: High costs per acquisition due to calling uninterested leads.
- Customer Fatigue: Brand erosion caused by irrelevant telemarketing contacts.
- Resource Misallocation: Missing high-value "hidden" subscribers while focusing on low-probability segments.

**Project Objectives**

The goal is to develop a robust classification pipeline capable of predicting subscription outcomes that identifies which customers are most likely to subscribe to a term deposit. The model analyzes a multi-dimensional feature set:

- Demographic Variables: Age, job type, marital status, and education.
- Financial Variables: Yearly balance, housing/personal loan status, and credit defaults.
- Campaign Interactions: Contact type, timing (month/day), and results of previous outreach.
- Economic Factors: Employment variation rates, consumer price indices, and Euribor interest rates.

**Business Problem**

Direct marketing campaigns are expensive. Calling every customer in the database results in:

- Low ROI: High operational costs for low conversion.
- Customer Fatigue: Annoying customers who have zero interest in the product.
- Resource Inefficiency: Sales agents spending time on "cold" leads.

### Project Summary: Term Deposit Subscription Prediction

**Objective**
The goal of this project was to build a predictive model to identify customers most likely to subscribe to a bank term deposit. Given the high cost of manual outreach, the focus was on maximizing Recall (identifying as many potential subscribers as possible) while maintaining a high F1-Score to ensure call center efficiency.

**Technical Workflow**
Data Engineering: Engineered features like call_efficiency and total_contacts, which proved to be top predictors of success.

Addressing Imbalance: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to overcome the natural "class imbalance" where most customers say "no."

Model Selection: Evaluated four models (Logistic Regression, KNN, Random Forest, and XGBoost). XGBoost was selected as the "Champion Model" after optimization via RandomizedSearchCV.

Interpretability: Utilized SHAP values to decode the "Black Box," revealing that call duration, previous campaign success, and the absence of existing loans are the strongest indicators of subscription.

**Final Model Performance**
The optimized XGBoost model achieved a Recall of ~84%, meaning the bank can now identify 8 out of every 10 potential subscribers. By prioritizing leads with high probability scores, the bank can significantly reduce "wasted" calls and increase the overall ROI of the marketing campaign.

**Strategic Recommendations**
Lead Prioritization: The sales team should focus exclusively on high-probability leads identified by the model, specifically those with high call_efficiency scores.

Timing is Key: Marketing efforts should be intensified in months like June and August, while avoiding May, which showed a negative correlation with success in this dataset.

Customer Profile: Targeting customers who do not currently have housing or personal loans will yield a higher conversion rate, as these individuals appear to have more liquid capital for term deposits.

**Tech Stack**

Language: Python 3.11

Libraries: Pandas, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), SHAP

Visualization: Seaborn, Matplotlib, Plotly

# The Data Science Workflow

1. Data Collection & Cleaning:

     Aggregating historical data from the UCI Bank Marketing Dataset.
   
     Data link: https://archive.ics.uci.edu/dataset/222/bank+marketing

2. Exploratory Data Analysis (EDA):

   - Analyzed demographic features (age, job, marital status) against the target.
   - Identified Class Imbalance: 88% "No" vs. 12% "Yes".
   - Data Cleaning Strategy: Data-Driven Decisions(after performing initial EDA concluded )
     - Statistical Validation: Performed Chi-Square tests on categorical features.
     - Key Finding: Despite high "unknown" rates in poutcome (81%) and contact (28%), p-values ($< 0.05$) confirmed they are significant predictors of           subscription.
     - Action: Retained "unknown" as a valid category for significant features; used mode-imputation for features with $< 5\%$ unknowns (job, education).
   - Detected Data Leakage: Identified that the duration feature must be handled carefully to ensure the model remains "predictive" rather than "descriptive."
     
3. Feature Engineering & Preprocessing

    - Encoding categorical variables and scaling numerical distributions to ensure model stability.
        - Categorical Encoding: One-Hot Encoding for nominal variables.
        - Scaling: Applied RobustScaler to numerical features (balance, age) to mitigate the impact of outliers.
      
4. Oversampling:
   
    Handling Imbalance: Utilizing SMOTE (Synthetic Minority Over-sampling Technique) to address the minority class (subscribers).(Used SMOTE to balance the training set, ensuring the model learns the characteristics of the "Yes" class.)
   
5. Model Development:

   Benchmarking KNN/Logistic Regression, Random Forest, and XGBoost to find the optimal balance of precision and recall.

   - Baseline: KNN and Logistic Regression
   - Ensemble: Random Forest
   - Gradient Boosting: XGBoost (Optimized via RandomizedSearchCV)

6. Evaluation Metrics

    Primary Metric: F1-Score & PR-AUC (due to class imbalance).

    Business Metric: Lift/Gain charts to show the percentage of subscribers captured within the top 20% of the customer list.

7. **Model Interpretability (SHAP)**

    Instead of a "Black Box," this project uses SHAP (SHapley Additive exPlanations) to show exactly which features influenced each prediction, ensuring the model is transparent and ready for banking
    compliance. Implementing SHAP values to explain why the model predicts a "Yes" for specific customer profiles.)

**Key Performance Metrics**

We prioritize metrics that reflect business value over simple accuracy:

- Recall (Sensitivity): Ensuring we don't miss potential subscribers.
- Precision: Minimizing "wasted" calls to non-subscribers.
- F1-Score: The harmonic mean to balance precision/recall in our imbalanced dataset.
- Lift/Gain Charts: Measuring how much better the model performs compared to a random selection strategy.

**Key Insights**

  Contact Month: Campaigns in May had the highest volume but the lowest conversion rate.

  Previous Success: Customers who subscribed previously are 6x more likely to subscribe again.

  Economic Context: The euribor3m rate showed a strong inverse correlation with subscription probability.

**Expected Business Benefits**

- Enhanced Conversion: Identifying high-propensity clients early for tailored offers.
- Resource Optimization: Focusing sales teams on leads with the highest probability of success.
- Informed Strategy: Using model insights to design campaigns that align with current economic indicators (e.g., interest rate sensitivity).

**Risks & Mitigation**

- Data Privacy: Ensuring all PII (Personally Identifiable Information) is handled according to regulatory standards.
- Model Drift: Mitigated by suggesting a retraining schedule as macroeconomic conditions change.
- Duration Bias: Crucial Note: The duration feature is excluded from predictive training to avoid data leakage, ensuring the model is usable before a call is made.

**One-Click Portfolio:**
  "Just run python run_all.py and it will build the entire project from scratch." - working on it

**Folder Structure**

bank-subscription-propensity/
├── data/               # Raw and processed data (GitIgnored)
├── images/             # Plots for your README (SHAP, Confusion Matrix)
├── notebooks/          # Exploratory Data Analysis and Experiments
│   └── 01_eda_and_modeling.ipynb
├── src/                # Modular Python scripts (Optional but pro)
│   ├── preprocessing.py
│   └── model_trainer.py
├── .gitignore          # Files GitHub should ignore (like large datasets)
├── README.md           # The professional documentation we wrote
└── requirements.txt    # Library dependencies

**How to Run**

Clone the repo: git clone ...

Install requirements: pip install -r requirements.txt

Run the notebook: jupyter notebook main.ipynb

