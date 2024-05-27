# Metalearners for Estimating Individual Treatment Effects

This repository contains a comprehensive tutorial on metalearners for estimating individual treatment effects (ITEs). The tutorial is inspired by the seminal paper "Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning" by Kunzel et al. (2019). It covers the implementation and experimentation with various meta-algorithms including S-learners, T-learners, and X-learners, using the Decision Tree Regressor from the scikit-learn Python library as the base learner.

## Overview
The project aims to estimate ITEs, which are crucial for understanding the effect of a treatment at an individual level. Since we can never observe both potential outcomes for the same individual, the goal is to estimate these effects using observed data.

## Key Components

### 1. Data Generation Process (DGP) for Simulations
- The tutorial includes a detailed simulation process to generate sample data with various features, treatment assignments, and potential outcomes.
- It also defines confounders and calculates the true conditional average treatment effect (CATE) for evaluation purposes.

### 2. Implementing T-, S-, and X-learners
- **T-Learner**: Trains separate models for the treated and control groups, estimating ITEs as the difference between these models' predictions.
- **S-Learner**: Uses a single model that includes the treatment indicator as a feature, estimating ITEs by comparing predictions for treated and untreated scenarios.
- **X-Learner**: Combines elements of both T- and S-learners, refining estimates through an iterative process that involves predicting residuals and weighting functions.

### 3. Experimentation with Sample Size and Class Imbalance
- The tutorial explores how different sample sizes and treatment proportions affect the performance of each learner.
- It provides code for running simulations with various configurations, analyzing the results in terms of mean squared error (MSE), bias, and variance.

### 4. Evaluation Metrics
- **Mean Squared Error (MSE)**: Measures the average squared difference between observed outcomes and predictions.
- **Bias and Variance**: Analyzes the error introduced by approximations and the stability of model predictions across different datasets.

### 5. Summary and Conclusion
- The tutorial concludes with a comparative analysis of the three learners, highlighting the strengths and weaknesses of each approach based on simulation results.
- It emphasizes the effectiveness of the X-learner in handling complex data structures and varying treatment effects.

## How to Use This Repository
- Clone the repository and ensure you have the necessary Python packages installed using the `requirements.txt` file. Run the following command:
  ```bash
  pip install -r requirements.txt
  ```

- Follow the step-by-step instructions in the Jupyter Notebook to understand the implementation details and run the provided simulations.

By following this tutorial, you will gain a solid understanding of how to estimate individual treatment effects using advanced meta-learning techniques and the impact of sample size and class imbalance on these estimations.
