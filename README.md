# Titanic Survival Prediction using Machine Learning

## Project Overview

This project uses machine learning techniques to predict whether a passenger survived the Titanic disaster. It covers the complete machine learning workflow, including data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and prediction.

## Dataset

The project uses the Titanic dataset from Kaggle. The dataset contains information about passengers, such as their age, gender, ticket fare, passenger class, family members traveling with them, and survival status.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Project Workflow

The following steps were carried out during the project:

1. Loaded and explored the dataset.
2. Identified and handled missing values.
3. Performed exploratory data analysis (EDA) to understand the data.
4. Created new features such as **Family Size** and **Alone**.
5. Converted categorical values into numerical values.
6. Split the dataset into training and testing sets.
7. Trained multiple machine learning models.
8. Compared the performance of each model.
9. Built a prediction system to estimate passenger survival based on user input.

## Machine Learning Models

The following classification algorithms were implemented and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)

## Model Performance

  Model                          Accuracy 

 Logistic Regression              81.01% 
 Decision Tree                    78.21% 
 Random Forest                    84.36% 
 K-Nearest Neighbors              72.07% 

## Best Performing Model

Among the four algorithms, the **Random Forest Classifier** achieved the highest accuracy of **84.36%**, making it the best-performing model for this project.

## Features Used for Prediction

* Sex
* Age
* Fare
* Embarked
* Alone
* Family Size
* Passenger Class (Pclass)

## Conclusion

This project helped me understand the complete machine learning pipeline, from preparing raw data to building, evaluating, and comparing multiple classification models. It also provided hands-on experience with feature engineering, model evaluation, and creating an interactive prediction system.
