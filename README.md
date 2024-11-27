# Diabetes Prediction based on diagnostic measures
## Project Overview
This project focuses on developing and evaluating machine learning models to predict the presence of diabetes based on patient data. The primary objective was to build a reliable classification system while analyzing key features influencing diabetes prediction.

## Data Sources
The dataset used in this project is based on a diabetes diagnostic dataset from Kaggle . It includes demographic, physiological, and medical measurements for patients, allowing us to predict diabetes outcomes (1 for diabetes, 0 for no diabetes).

Key columns include:

- Demographic Features: Age, Pregnancies.
- Physiological Features: Glucose, BloodPressure, SkinThickness, Insulin, BMI.
- DiabetesPedigreeFunction: Represents a patient’s likelihood of diabetes based on genetic predisposition.
- Outcome: Target variable indicating whether the patient has diabetes (1) or not (0).

## Project Structure
1. Data Preparation:

The raw dataset was preprocessed to handle missing values and ensure consistency:
- Missing values were imputed using median or mean for continuous variables.
- Outliers and zeros in physiological measurements were analyzed and appropriately handled.
- Feature scaling was applied to ensure uniformity for machine learning algorithms requiring normalized input.
- Categorical features, such as age groups, were created and encoded using one-hot encoding for better model interpretability.

2. Feature Engineering:

Several engineered features were introduced to improve model performance:
- Lagged transformations for skewed data columns like Insulin and SkinThickness.
- Age Group Binning: Age was grouped into meaningful ranges (e.g., 21–30, 31–40) to capture patterns related to diabetes.
- Scaling Continuous Variables: Features like Glucose and BMI were standardized to improve model convergence.

3. Model Selection and Training:

Six classification models were trained and optimized:
- **Logistic Regression**
- **Ridge Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **SVM (Support Vector Machine)**
- **KNN (K-Nearest Neighbors)**

Hyperparameter tuning for each model was conducted using GridSearchCV to achieve optimal performance.

4. Model Evaluation:

- The models were evaluated using metrics such as accuracy, precision, recall, F1-Score, and confusion matrices.
- Cross-validation was implemented to ensure robust evaluation and prevent overfitting.

5. Feature Importance Analysis:

- Feature importance analysis was conducted using Random Forest and XGBoost models to understand the most critical predictors of diabetes.
- Key features included Glucose, BMI, Age, and DiabetesPedigreeFunction, which consistently showed strong predictive power.

6. Final Model Selection:

- The **K-Nearest Neighbors (KNN)** model was selected as the best-performing model based on its high recall and balanced F1-Score, crucial for minimizing false negatives in medical contexts.

## Key Insights and Findings
1. Best Model Performance:

The KNN Classifier achieved the best balance between recall and precision:
- Test Recall: 90.91% (ensures most diabetes cases are correctly identified).
- Test F1-Score: 66.67% (balanced measure of precision and recall).
- This makes KNN particularly suitable for medical applications where identifying at-risk patients is critical.

2. Feature Importance:

Key predictors of diabetes included:
- Glucose: Strongest predictor with consistent importance across models.
- BMI: Indicates obesity-related risk factors for diabetes.
- Age: Higher age groups showed increased diabetes prevalence.
- DiabetesPedigreeFunction: Genetic predisposition had a moderate influence on diabetes prediction.

3. Trade-Offs:

KNN’s high recall comes at the cost of slightly lower precision, meaning more false positives are generated. However, in healthcare, prioritizing recall ensures fewer cases are missed, reducing risks for patients.

## License

[MIT](https://choosealicense.com/licenses/mit/)