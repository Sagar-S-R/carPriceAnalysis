# Car Price Classification using Machine Learning

## Overview
This project applies multiple machine learning classification algorithms to predict the number of previous owners of a car based on its features. The dataset contains various attributes such as brand, model, year, engine size, fuel type, transmission type, mileage, number of doors, and price.

## Dataset Information
- **Number of records:** 10,000
- **Features:**
  - `Brand` (Categorical)
  - `Model` (Categorical)
  - `Year` (Integer)
  - `Engine_Size` (Float)
  - `Fuel_Type` (Categorical)
  - `Transmission` (Categorical)
  - `Mileage` (Integer)
  - `Doors` (Integer)
  - `Owner_Count` (Integer, Target Variable)
  - `Price` (Integer)

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Data Preprocessing
- Handling missing values (if any)
- Encoding categorical variables
- Splitting dataset into training and testing sets (80% train, 20% test)
- Scaling numerical features using `StandardScaler`

## Models Implemented
1. **Logistic Regression**
2. **Support Vector Machine (SVM) with GridSearch for Hyperparameter Tuning**
3. **K-Nearest Neighbors (KNN) with GridSearch for Best Neighbors Selection**
4. **Decision Tree with GridSearch for Optimal Depth**
5. **Random Forest with GridSearch for Best Parameters**

## Performance Evaluation
- **Metrics Used:**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - ROC Curve (where applicable)

- **Model Comparison:** A bar plot is generated to compare the accuracies of different models visually.

## How to Run the Project
1. Install required dependencies:
   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Place the dataset (`car_price_dataset.csv`) in the project directory.
3. Run the Python script or Jupyter Notebook containing the machine learning models.

## Results & Observations
- The Decision Tree and Random Forest models showed slightly better accuracy compared to other models.
- Logistic Regression and SVM performed poorly due to the complex, multi-class nature of the target variable.
- Feature scaling significantly improved KNN performance.

## Future Improvements
- Implement feature engineering to extract more meaningful insights.
- Use advanced ensemble methods like Gradient Boosting or XGBoost.
- Perform hyperparameter tuning with more parameters for better optimization.
- Apply deep learning techniques for potentially improved classification accuracy.


## Author
- **Your Name**
- **GitHub:** [Your GitHub Profile](https://github.com/Sagar-S-R)


