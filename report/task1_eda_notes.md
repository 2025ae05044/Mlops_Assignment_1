# Task 1: Data Acquisition and Exploratory Data Analysis

## Dataset

The processed Cleveland Heart Disease dataset was downloaded from the UCI Machine Learning Repository. The downloaded data contains 303 records and 14 columns, comprising 13 input features and one target.

## Data Cleaning

The original missing-value markers were converted to missing values. The original target values were converted into a binary target, where 0 represents the absence of heart disease and 1 represents its presence.

The cleaned dataset contains 303 records, 14 columns and no duplicate rows. Four missing values were observed in `ca` and two in `thal`. These feature values were retained for handling in the model preprocessing pipeline.

## Class Distribution

- No heart disease (target 0): 164 records (54.1%).
- Heart disease present (target 1): 139 records (45.9%).

The two classes are reasonably close in size, although model evaluation should still consider precision, recall and ROC-AUC rather than accuracy alone.

## Exploratory Data Analysis

Feature histograms were generated for age, resting blood pressure, cholesterol, maximum heart rate and oldpeak. A correlation heatmap was created to examine relationships among numeric columns. A class-balance chart was generated to visualize the binary target distribution.

The exact distribution shapes and strongest correlations should be described after inspecting the generated plots. Correlations involving integer-coded categorical features require cautious interpretation.

## Conclusion

The dataset was successfully downloaded, cleaned and saved as a CSV file. Initial data validation and the required EDA visualizations were prepared for subsequent feature engineering and model development.
