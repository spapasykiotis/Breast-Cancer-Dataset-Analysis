
# Breast Cancer Dataset Analysis using WEKA

## Project Overview

This project focuses on analyzing the Breast Cancer dataset using machine learning algorithms in WEKA. The analysis includes both classification and clustering techniques to predict recurrence events of breast cancer based on various attributes. The classification task uses the J48 decision tree classifier, while the clustering task employs the SimpleKMeans algorithm.

### Dataset Information
The dataset is sourced from [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/mahima5598/breast-cancer-datasetnodecaps) and contains the following attributes:

- **class**: {no-recurrence-events, recurrence-events}
- **age**: {20-29, 30-39, 40-49, 50-59, 60-69, 70-79}
- **menopause**: {premeno, ge40, lt40}
- **tumor-size**: {0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54}
- **inv-nodes**: {0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 24-26}
- **node-caps**: {no, yes}
- **deg-malig**: {1, 2, 3}
- **breast**: {left, right}
- **breast-quad**: {central, left_low, left_up, right_low, right_up}
- **irradiat**: {no, yes}

### Dataset License
This dataset is taken from [Kaggle](https://www.kaggle.com/datasets/mahima5598/breast-cancer-datasetnodecaps) and is available under the appropriate usage license provided by Kaggle. Please ensure you follow Kaggle's terms and conditions when using this dataset.

## Algorithms Used

### 1. J48 Classifier (Decision Tree)

- **Algorithm**: J48 is a decision tree algorithm, which is an implementation of the C4.5 algorithm. It is used for classification purposes to predict the recurrence of breast cancer.
- **Parameters**: The J48 classifier is used with its default parameters.

### 2. SimpleKMeans Clustering

- **Algorithm**: SimpleKMeans is used for clustering the dataset into groups based on the attributes, excluding the class attribute. This helps in identifying potential patterns and groups within the data.
- **Parameters**:
  - **Initialization Method**: k-means++ (for better initialization of centroids)

## Evaluation Metrics

The following metrics are used to evaluate the performance of the classification and clustering algorithms:

### Classification:
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positive instances.
- **F-Measure**: The harmonic mean of precision and recall.
- **Error Rate**: The proportion of incorrect predictions.

### Clustering:
- **Cluster Evaluation**: Displays clustering results, including information about the number of clusters, cluster sizes, and the performance of the clustering.

## Requirements
- Java 8 or higher
- WEKA 3.8 or higher
- Dataset (`breast-cancer-final.arff`) should be placed in the `data/` directory.
