# Car Price Classification using Machine Learning

## Overview
This project applies various machine learning classifiers to predict the `Owner_Count` of a car based on features like brand, model, year, engine size, fuel type, transmission, mileage, and number of doors. The dataset contains 10,000 entries and 10 features.

## Dataset
- **Source:** `car_price_dataset.csv`
- **Features:**
  - `Brand`, `Model` (Categorical)
  - `Year`, `Engine_Size`, `Mileage`, `Doors`, `Owner_Count`, `Price` (Numerical)
  - `Fuel_Type`, `Transmission` (Categorical)
- **Target Variable:** `Owner_Count`

## Dependencies
Ensure you have the following libraries installed before running the code:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Project Workflow
1. **Load Dataset:** Read and display the dataset.
2. **Exploratory Data Analysis (EDA):**
   - Summary statistics
   - Missing values check
   - Data visualization (distribution plots, correlation heatmaps)
3. **Feature Selection & Preprocessing:**
   - Drop non-numeric features for simplicity
   - Standardize numerical features
   - Train-test split (80-20 split)
4. **Model Training & Evaluation:**
   - Logistic Regression
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Hyperparameter tuning using `GridSearchCV`
5. **Comparison of Model Performances:**
   - Accuracy scores
   - Confusion matrices
   - Classification reports
   - Bar chart comparison

## Results
- Logistic Regression Accuracy: **~20.15%**
- SVM Accuracy: **~19.55%**
- KNN Accuracy: **~20.3%**
- Decision Tree Accuracy: **~20.6%**
- Random Forest (Best Model) tuned using `GridSearchCV`.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Place `car_price_dataset.csv` in the project directory.
3. Run the Python script:
   ```bash
   python main.py
   ```

## Future Improvements
- Encode categorical features (`Brand`, `Model`, `Fuel_Type`, `Transmission`)
- Try other models like XGBoost, Gradient Boosting, or Neural Networks
- Improve feature engineering

## Author
- **Your Name**
- **GitHub:** [Your GitHub Profile](https://github.com/Sagar-S-R)


