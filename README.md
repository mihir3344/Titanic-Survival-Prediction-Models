# Titanic Survival Prediction

This repository contains machine learning models used for predicting the survival of passengers aboard the Titanic. Several classification algorithms were tested and evaluated to find the best-performing model for this task.

## Objective
The objective of this project is to predict the survival of passengers on the Titanic based on various features such as age, sex, class, and more. This could be used for educational purposes or to understand the prediction accuracy of different machine learning models.

## Key Features

- **Data Preprocessing**: Data was cleaned and processed to handle missing values, categorical features, and scaling where necessary.
- **Modeling**: Multiple machine learning classification models were tested, including Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Random Forest, and others.
- **Hyperparameter Tuning**: Optimized model performance using Grid Search with cross-validation.
- **Evaluation Metrics**: Performance was evaluated using metrics such as accuracy, precision, recall, F1-score, and classification reports.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Model Performance

The following table summarizes the performance of various classification models used in the Titanic survival prediction task. The metrics include **Accuracy**, **Precision**, **Recall**, and **F1-Score** for both classes (0: Died, 1: Survived).

| **Model**                         | **Accuracy** | **Precision (0)** | **Recall (0)** | **F1-Score (0)** | **Precision (1)** | **Recall (1)** | **F1-Score (1)** | **Macro avg** | **Weighted avg** |
|-----------------------------------|--------------|-------------------|----------------|------------------|-------------------|----------------|------------------|----------------|-------------------|
| **Logistic Regression L2**        | 0.8358       | 0.88              | 0.87           | 0.88             | 0.75              | 0.77           | 0.76             | 0.82           | 0.84              |
| **Logistic Regression L1**        | 0.8358       | 0.89              | 0.87           | 0.88             | 0.74              | 0.78           | 0.76             | 0.81           | 0.84              |
| **K-Nearest Neighbors**           | 0.8545       | 0.88              | 0.90           | 0.89             | 0.79              | 0.77           | 0.78             | 0.84           | 0.85              |
| **Support Vector Classifier**     | 0.8507       | 0.88              | 0.90           | 0.89             | 0.79              | 0.76           | 0.77             | 0.83           | 0.85              |
| **Linear SVC**                    | 0.8396       | 0.89              | 0.87           | 0.88             | 0.75              | 0.78           | 0.77             | 0.82           | 0.84              |
| **Random Forest**                 | 0.8433       | 0.89              | 0.88           | 0.88             | 0.76              | 0.78           | 0.77             | 0.83           | 0.84              |
| **Decision Tree**                 | 0.8060       | 0.88              | 0.82           | 0.85             | 0.69              | 0.78           | 0.73             | 0.80           | 0.81              |
| **AdaBoost**                      | 0.8284       | 0.90              | 0.84           | 0.87             | 0.72              | 0.81           | 0.76             | 0.82           | 0.83              |
| **Gaussian Naive Bayes**          | 0.6679       | 0.67              | 1.00           | 0.80             | 1.00              | 0.01           | 0.02             | 0.51           | 0.67              |
| **Multi-layer Perceptron (MLP)**  | 0.8396       | 0.88              | 0.88           | 0.88             | 0.76              | 0.77           | 0.76             | 0.82           | 0.84              |

## Key Insights:
- **K-Nearest Neighbors (KNN)** performed the best with an accuracy of **85.45%** and strong F1-scores across both classes.
- **Support Vector Classifier (SVC)** closely followed KNN with an accuracy of **85.07%**, performing well on both classes.
- **Logistic Regression L2 and L1** models achieved **83.58%** accuracy, with good precision for class 0 (Died) and reasonable recall for class 1 (Survived).
- **Random Forest** also provided strong results with **84.33%** accuracy, showing good balance between precision and recall.
- **Gaussian Naive Bayes** showed poor performance, especially for class 1 (Survived), due to an extremely low recall value.

## Conclusion:
The **K-Nearest Neighbors (KNN)** and **Support Vector Classifier (SVC)** models are the best performing, providing high accuracy and balanced precision-recall scores. The **Random Forest** model also performs well, making it a reliable choice for Titanic survival prediction. **Logistic Regression** and **Linear SVC** models show good results, though their accuracy is slightly lower compared to tree-based methods. **Gaussian Naive Bayes** performs poorly, especially in predicting survivors.

## Future Work:
- Exploring other machine learning algorithms and ensembling methods to improve the prediction accuracy further.
- Hyperparameter tuning using Grid Search or Randomized Search to optimize model performance.

## Technologies Used:
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

Feel free to explore the repository, experiment with different models, and contribute to improving the Titanic survival prediction model.

