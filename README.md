**Problem Framing:**
- The problem can be framed as a binary classification task where the goal is to predict which customers are more likely to convert (respond positively to the term deposit offer) and which are not likely to convert and build a machine learning model to classify customers into these two groups.

**Goal:**
- The goal is to build a binary classifier such that it has high precision (ie low false positives) which would reduce the number of "wasted calls" (calls to non-converting customers).


**About dataset:**
- The file bank-additional-full.csv contains 41176 records out of which have 4639 customer records (11%) have Term_deposit = YES and 36537 customer records (89%) have Term_deposit = NO


**Steps taken to solve the challenge:**

**1. Exploratory Data Analysis**

- The file bank-additional-full.csv is converted into pandas dataframe
- The CSV file have 20 input features and output binary variable "y" which denotes Term_deposit
- The target class "y" is highly imbalanced. 89% of records have y=0 and 11% of records have y=1
- Feature: Campaign
    - About 42% of the customers are called only once.
    - The percentage of customers who have subscribed to Term_deposit where campaign=1 is 2% higher than the customers where campaign is between 2 to 3
- Feature: pdays
    - Around 96% of customers were not contacted from the last campaign.
    - The chances of a customer to subscribe to Term_deposit are relatively high within the first 3 weeks up to 20 days; further we have high variance and small data.
    - People, who was not being previously contacted, have the lowest probability to subscribe to Term_deposit (9%)
- Feature: previous
    -  Around 90% of the customers were not contacted before this campaign.
    -  It is observed that more previous contacts mean more chances to subscribe to the Term_deposit, but only to some degree. For eg. if the customer has been contacted 5 times previously before the current campaign then there is 72% chance that the customer will subscribe to the Term_deposit
- Feature: poutcome
    - If a customer has subscribed to previous campaign, then they are more likely to subscribe to the Term_deposit during the current one.
- Feature: contact
    - Most people were contacted by the cellular about 63%
    - People contacted via cellular are relatively more likely to agree to subscribe to the term deposit.
- Feature: age
    *   Median age: 38 y.o.
    *   Mean age: 40 y.o.
    *   People older than 65 y.o. are more likely to agree to subscribe a bank term deposit (40 to 50%).
    * Also about 45% of people between 15-20 y.o. are likely to subscribe to Term deposit
- Social and Economic context attributes
    - If employment rate, euribor 3 month rate (The Euro Interbank Offered Rate) and number of employees are varying to a larger extent in the given time period, then people are less likely to subscribe to Term_deposit.

**2. Data preprocessing and Feature selection and Feature encoding:**

- Feature selection:
  - Different feature selection techniques like ANOVA and Chi2 are experimented. Low variance features are dropped. The feature "euribor 3 month rate" is more than 97% correlated with employment rate so euribor3m is dropped.
- Feature Encoding:
  - Education has ordered categorical values and these values are encoded in an order from university degree untill illiterate. 
  - OneHot Encoding is used to encode the categorical features - 'job', 'marital', 'contact', 'month', 'poutcome'. 
  - The Binary valued features - 'default', 'housing', 'loan' are also encoded
- Data preprocessing:
  - The 'unknown' values are treated as separate feature values in the model. This is because these values may not be random and may themselves be information.
  - It is given that the feature, "duration", should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model. Hence duration is dropped.
  - Data is split into training and test set.
  - Standard scaling is applied to the data.

**3. Modelling**

- The problem is framed as a binary classification task and the goal is to build a binary classfier which can predict if a customer is likely to subscribe to a Term_deposit based on the given input features
- Since the target class is imbalanced, the data is sampled using oversampling, undersampling and hybrid approaches.
- Different methods to tune hyperparameters such as GridSearchCV, RandomSearchCV and BayesSearchCV is applied together with Stratified K-fold cross validation and BayesSearchCV is found to be the best approach for tuning hyper parameters.
- Logistic Regression:
    - LR model is developed on different types of sampled data. The best LR model was with undersampling approach using TomekLinks that yielded a ROC_AUC score of 79% and Precision of 70%
- Random Forest:
    - Random Forest model is developed on different types of sampled data and the best model (undersampling using TomekLinks) yielded a ROC_AUC score of 77% and precision 58%
- XGBoost
    - XGBoost model is developed on different types of sampled data and the best model (with feature selection applied)
    yielded a ROC_AUC score of 81% and precision of 86%
- Neural Networks and One class SVM approaches were also tried but it is observed that XGBoost performs better in this case. Best neural net model (with class weight sampling) yielded a ROC_AUC score of 79% and precision of 38%

- Results from the model is shown in the below table. Best model is highlighted in bold

<table>
  <tr>
    <th>Model</th>
    <th>ROC_AUC</th>
    <th>Precision</th>
    <th>Recall</th>
  </tr>
  <tr>
    <td>Logistic Regression (hybrid sampling - SMOTETomek)</td>
    <td>79%</td>
    <td>60%</td>
    <td>30%</td>
  </tr>
   <tr>
    <td>Logistic Regression (hybrid sampling - SMOTEENN)</td>
    <td>79%</td>
    <td>46%</td>
    <td>56%</td>
  </tr>
  <tr>
    <td>Logistic Regression (under sampling - TomekLinks)</td>
    <td>79%</td>
    <td>70%</td>
    <td>21%</td>
  </tr>
  <tr>
    <td>Random Forest (hybrid sampling - SMOTETomek)</td>
    <td>78%</td>
    <td>49%</td>
    <td>36%</td>
  </tr>
  <tr>
    <td>Random Forest (under sampling - TomekLinks)</td>
    <td>77%</td>
    <td>58%</td>
    <td>28%</td>
  <tr>
    <td>XGBoost (hybrid sampling - SMOTETomek)</td>
    <td>78%</td>
    <td>57%</td>
    <td>33%</td>
  </tr>
  <tr>
    <td>XGBoost (over sampling and cleaning - SMOTEENN)</td>
    <td>79%</td>
    <td>44%</td>
    <td>58%</td>
  </tr>
  <tr>
    <td>XGBoost (Recursive Feature Elimination)</td>
    <td>81%</td>
    <td>84%</td>
    <td>10%</td>
  </tr>
  <tr>
    <td><b>XGBoost (after Feature Selection using multiple techniques)</td>
    <td><b>81%</td>
    <td><b>86%</td>
    <td><b>11%</td>
  </tr>
  <tr>
    <td>Neural Network (with class weight sampling)</td>
    <td>79%</td>
    <td>39%</td>
    <td>61%</td>
  </tr>
  <tr>
    <td>Neural Network (with PCA)</td>
    <td>69%</td>
    <td>38%</td>
    <td>31%</td>
  </tr>
</table>


**Drivers of conversion**
- Feature importance is calculated for the XGBoost model and the top 5 features that contribute to the target are found to be: nr.employed, emp.var.rate, cons.conf.idx, poutcome_success, month


**Observations and Recommendations**

- Observations:
  - Among all models, XGBoost classifier has the highest ROC_AUC score and precision
  - Based on the goal, the important performance metric to optimise in order to reduce wasted calls and improve cost-efficiency is "Precision".
  - Precision is a measure of how many of the calls made actually result in conversions. It is calculated as the ratio of the true positive calls (calls that lead to conversions) to the total calls made (true positive + false positive calls). 
  - In the context of this case, reducing wasted calls means making sure that when a call is made, it is more likely to result in a conversion. High precision means a high percentage of calls made leading to actual conversions.

- Recommendations:
  - In order to save calls, precision needs to be increased. 
  - Precision of the XGBoost classifier can further be increased by setting a threshold on the model's predicted probabilities to determine which customers to call. 
  - The threshold can be adjusted based on our trade-off between reducing wasted calls and not missing potential conversions.
  - By optimizing precision, we aim to make the marketing campaign more cost-efficient by reducing the number of calls made to non-converting customers. This approach helps us to achieve our goal of losing as little business as possible while minimizing the cost of calls.


**Cost efficiency analysis**

- The best performing XGBoost model is taken and cost efficiency analysis is performed based on the given data and the results are as follows
  - Profit generated from successful calls = 8240.00 Euros
  - Calling_cost = 960.00 Euros
  - Overall_profit = 7280.00 Euros
  - Wasted call percentage = 14.17%

**Conclusion**

- XGBoost model outperforms other models for the given task in identifying customers who has the highest potential to subscribe to Term_deposit.
- The model is optimised for precision as the goal is to reduce the number of calls by predicting the "wasted calls" (calls to non-converting customers).
- Threshold for predicted probabilities can be increased accordingly to reduce wasted calls. On a side note, if the threshold is decreased, the recall score would improve which would reduce the number of missed opportunities.
- Depending on the target business KPI (e.g. reduce call, increase sales) the threshold can be adjusted accordingly
- Based on the predicted Term_deposit probabilities of the customers, it is recommended to call the customers with higher probabilities in the first cohort


**ML system design and End-to-End implementation of ML model**

- Since the goal of this case study is to experiement with different approaches
and come up with the best performing ML model, the code is written in jupyter notebook for easy readability and visualisation purposes.

- Once the model is finalised, the production ready code shall have defined  modules for each functionality.

- Before beginning with modelling, it is important to define the following:
  - Project goal
  - Target KPI
  - ML evaluation metrics that needs to be optimised in order to acheive the target goal
  - Performance constraints (acceptable tolerance for errors)
  - Personalisation of ML models and
  - Project constraints (resources and timeline)

- It is important to have a data strategy and data governance framework in place which can take care of data security and privacy, bias in the data, data quality and manage the data effectively.

- The end to end pipeline shall have the following modules:
 Consume data from source-> transform data (if required) -> perform data quality checks -> Data preprocessing and Feature engineering -> ML model development and Hyper parameter tuning -> ML model selection  -> ML model training -> ML model evaluation -> ML model prediction -> ML model interpretability -> ML model deployment -> ML model serving via API. 
- Please find attached High level architecture diagram for a visual overview 

- MLOps framework can be implemented to take care of model versioning , CI/CD in cloud. For e.g. orchestration tools like Airflow can take of automatic scheduling of the data and ML pipeline, model monitoring, model retraining, checking for data drift and model drift over time, model evaluation, load balancing incoming requests to the API etc.,


Please find the detailed observation, experiments and results in the notebook file Bank_marketing_case_study.ipynb 
