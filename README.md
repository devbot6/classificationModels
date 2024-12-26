# Radar Signal Classification: From SVM to Decision Tree

This repository explores the implementation of machine learning models for radar signal classification, focusing on the transition from using a Support Vector Machine (SVM) to a Decision Tree classifier to accommodate multiple classification categories.

## Project Overview

Radar signal classification is a complex task requiring robust and flexible algorithms to differentiate between multiple signal types effectively. This project documents the journey of identifying and implementing the most suitable machine learning model for this purpose.

## Models Implemented

### 1. Support Vector Machine (SVM)
- **Description**: Initially, an SVM was implemented to classify radar signals.
- **Reason for Selection**: SVMs are powerful tools for binary classification tasks due to their ability to find the optimal hyperplane that separates two classes effectively.
- **Limitation**: SVMs are primarily designed for binary classification. While multiclass SVMs exist, they are less flexible and efficient for scenarios requiring classification across multiple categories.
- **Outcome**: After testing, it was evident that the fixed binary approach of SVMs was unsuitable for the dynamic, multi-class requirements of radar signal classification.

### 2. Decision Tree
- **Description**: A Decision Tree classifier was implemented to address the limitations observed with SVMs.
- **Reason for Selection**: Decision Trees can handle multi-class classification tasks, making them a better fit for scenarios requiring flexibility in the number of classes.
- **Advantages**:
  - Handles multiple categories effectively.
  - Provides interpretability by visualizing the tree structure.
  - Can adapt to dynamic requirements as the number of categories changes.
- **Outcome**: The Decision Tree model demonstrated better adaptability and accuracy for radar signal classification tasks.

## Features
- Code implementation for both SVM and Decision Tree classifiers.
- Visualization of results and decision boundaries (where applicable).
- Comprehensive comparison of model performances.

## Prerequisites
- Python 3.7 or later
- Libraries:
  - `scikit-learn`
  - `numpy`
  - `matplotlib`
  - `pandas`
