"""
Run the main function at the end to estimate the CATE using T, S, and X-Learner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def generate_synthetic_data(n: int = 5000,
                            treatment_proportion: float = 0.1,
                            seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data for the Causal Inference project
    Args:
        n (int): The number of samples to generate
    """

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate features
    age = np.random.randint(18, 65, n)
    income = np.random.normal(50000, 10000, n)
    gender = np.random.choice(['Male', 'Female'], n)
    loyalty_program = np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6])

    # Define confounders that have no direct influence on the outcome
    confounder_1 = np.random.choice(['TypeA', 'TypeB', 'TypeC'], n)
    confounder_2 = np.random.normal(0, 1, n)

    # Define treatment assignment probabilities based on features
    prob_treatment = 0.2 + 0.1 * (age > 40) + 0.1 * (income > 60000) + 0.1 * (gender == 'Female') + 0.2 * (loyalty_program == 'Yes')

    # Add small perturbations (random noise/errors) to treatment probabilities
    prob_treatment += np.random.normal(0, 0.02, n)

    # Adjust probabilities to achieve the desired treatment proportion
    adjustment_factor = treatment_proportion / prob_treatment.mean()
    prob_treatment *= adjustment_factor
    prob_treatment = np.clip(prob_treatment, 0, 1)  # Ensure probabilities are between 0 and 1

    # Assign treatment based on probabilities
    treatment = np.random.binomial(1, prob_treatment)

    # Define potential outcomes based on features and treatment
    # Effect of control
    y0 = 10 + 0.01 * age + 0.00005 * income - 2 * (gender == 'Female') + 5 * (loyalty_program == 'Yes') + np.random.normal(0, 1, n)
    # Effect of treatment
    y1 = 15 + 0.02 * age + 0.00008 * income + 2 * (gender == 'Female') + 8 * (loyalty_program == 'Yes') + np.random.normal(0, 1, n)

    # Assign observed outcomes based on treatment
    y = treatment * y1 + (1 - treatment) * y0

    # Calculate the true CATE for each individual
    # These values are never identifiable in practice
    true_cate = y1 - y0

    # Create a DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Gender': gender,
        'LoyaltyProgram': loyalty_program,
        'Confounder1': confounder_1,
        'Confounder2': confounder_2,
        'Treatment': treatment,
        'Outcome': y,
        'TrueCATE': true_cate
    })

    data = pd.get_dummies(data, columns=['Gender', 'LoyaltyProgram', 'Confounder1'], drop_first=True, dtype=int)

    return data



def split_data(data: pd.DataFrame,
               drop_treatment: bool = True
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and validation sets
    Args:
        data (pd.DataFrame): The data to split
    Returns:
        X_train (pd.DataFrame): The training data
        X_val (pd.DataFrame): The validation data
        y_train (pd.Series): The training labels
        y_val (pd.Series): The validation labels
    """

    if drop_treatment:
        X = data.drop(['Treatment', 'Outcome', 'TrueCATE'], axis=1)
    else:
        X = data.drop(['Outcome', 'TrueCATE'], axis=1)

    y = data['Outcome']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def fit_val_model(X_train: pd.DataFrame, 
                  y_train: pd.Series, 
                  X_val: pd.DataFrame, 
                  y_val: pd.Series):
    """
    Fit a decision tree model to the training data and evaluate it on the validation data
    Args:
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training labels
        X_val (pd.DataFrame): The validation data
        y_val (pd.Series): The validation labels
    Returns:
        results (pd.DataFrame): The results of the model evaluation
    """

    # Define the hyperparameters to search
    # max_depth: The maximum depth of the tree
    # min_samples_split: The minimum number of samples required to split an internal node
    # min_samples_leaf: The minimum number of samples required to be at a leaf node
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [15, 10, 5],
        'min_samples_leaf': [1, 3, 5]
    }

    results = {'model': [], 'train_mse': [], 'val_mse': []}

    # Create the model
    for d in param_grid['max_depth']:
        for s in param_grid['min_samples_split']:
            for l in param_grid['min_samples_leaf']:
                model = DecisionTreeRegressor(max_depth=d, 
                                                min_samples_split=s, 
                                                min_samples_leaf=l,
                                                random_state=42)
                model.fit(X_train, y_train)
                results['model'].append(model)

                # Evaluate the model using mean squared error
                train_mse = mean_squared_error(y_train, model.predict(X_train))
                val_mse = mean_squared_error(y_val, model.predict(X_val))

                results['train_mse'].append(train_mse)
                results['val_mse'].append(val_mse)

    # convert results to a DataFrame
    results = pd.DataFrame(results)

    return results

def retrain_best_model(results: pd.DataFrame, 
                       X_train: pd.DataFrame, 
                       y_train: pd.Series, 
                       X_val: pd.DataFrame, 
                       y_val: pd.Series
                       ) -> DecisionTreeRegressor:
    
    """
    Retrain the best model on the full training data.
    Args:
        results (pd.DataFrame): The results of the model evaluation
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training labels
        X_val (pd.DataFrame): The validation data
        y_val (pd.Series): The validation labels
    Returns:
        best_model (DecisionTreeRegressor): The retrained model
    """
    
    # Find the best model
    best_model = results.loc[results['val_mse'].idxmin()]['model']

    # Combine the training and validation data
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    # Retrain the best model on the full training data
    best_model.fit(X_train_full, y_train_full)

    return best_model


def t_learner_cate(data: pd.DataFrame, 
                    model_c: DecisionTreeRegressor,
                    model_t: DecisionTreeRegressor
                    ) -> pd.DataFrame:
    
    """
    Use the T-Learner approach to estimate the Conditional Average Treatment Effect (CATE)
    Args:
        data (pd.DataFrame): The data to predict on
        model_c (DecisionTreeRegressor): The model to predict the control outcome
        model_t (DecisionTreeRegressor): The model to predict the treatment outcome
    Returns:
        data (pd.DataFrame): The data with the predicted CATE
    """

    # Predict the control and treatment outcomes
    X = data.drop(['Treatment', 'Outcome', 'TrueCATE'], axis=1)
    data['ControlOutcome'] = model_c.predict(X)
    data['TreatmentOutcome'] = model_t.predict(X)

    # Calculate the CATE
    data['PredictedCATE'] = data['TreatmentOutcome'] - data['ControlOutcome']

    # Calculate the difference between the predicted and true CATE
    data['CATEError'] = data['PredictedCATE'] - data['TrueCATE']

    return data


def s_learner_cate(data: pd.DataFrame, 
                    model: DecisionTreeRegressor
                    ) -> pd.DataFrame:
    """
    Use the S-Learner approach to estimate the Conditional Average Treatment Effect (CATE)
    Args:
        data (pd.DataFrame): The data to predict on
        model (DecisionTreeRegressor): The model to predict the outcome
    """

    X = data.drop(['Outcome', 'TrueCATE'], axis=1)

    # Predict the treated and control outcomes
    X['Treatment'] = 1
    data['PredictedOutcomeTreated'] = model.predict(X)
    X['Treatment'] = 0
    data['PredictedOutcomeControl'] = model.predict(X)

    # calculate the predicted CATE
    data['PredictedCATE'] = data['PredictedOutcomeTreated'] - data['PredictedOutcomeControl']

    # calculate the difference between the predicted and true CATE
    data['CATEError'] = data['PredictedCATE'] - data['TrueCATE']

    return data


def x_learner_cate(data: pd.DataFrame,
                   model_c: DecisionTreeRegressor,
                   model_t: DecisionTreeRegressor
                   ) -> pd.DataFrame:
    """
    Use the X-Learner approach to estimate the Conditional Average Treatment Effect (CATE)
    Args:
        data (pd.DataFrame): The data to predict on
        model_c (DecisionTreeRegressor): The model to predict the control outcome
        model_t (DecisionTreeRegressor): The model to predict the treatment outcome
    Returns:
        data (pd.DataFrame): The data with the predicted CATE
    """

    # split the data into treatment and control groups
    data_treatment = data[data['Treatment'] == 1]
    data_control = data[data['Treatment'] == 0]

    # feed the treatment group data to the control group model
    X_t = data_treatment.drop(['Treatment', 'Outcome', 'TrueCATE'], axis=1)
    data_treatment['PredictedOutcomeControl'] = model_c.predict(X_t)

    # feed the control group data to the treatment group model
    X_c = data_control.drop(['Treatment', 'Outcome', 'TrueCATE'], axis=1)
    data_control['PredictedOutcomeTreated'] = model_t.predict(X_c)

    # compute the residuals
    data_treatment['Residual'] = data_treatment['Outcome'] - data_treatment['PredictedOutcomeControl']
    data_control['Residual'] = data_control['PredictedOutcomeTreated'] - data_control['Outcome']

    # Fit a model to predict residuals in the treatment group
    model_t_residual = LinearRegression()
    model_t_residual.fit(data_treatment.drop(['Treatment', 'Outcome', 'TrueCATE', 'Residual', 'PredictedOutcomeControl'], axis=1), 
                        data_treatment['Residual'])

    # Fit a model to predict residuals in the control group
    model_c_residual = LinearRegression()
    model_c_residual.fit(data_control.drop(['Treatment', 'Outcome', 'TrueCATE', 'Residual', 'PredictedOutcomeTreated'], axis=1), 
                        data_control['Residual'])
    
    # Get a copy of the original data
    data_x_learn = data.copy()

    # Predict the control and treatment outcomes
    data_x_learn['TreatmentOutcome'] = model_t_residual.predict(data.drop(['Treatment', 'Outcome', 'TrueCATE'], axis=1))
    data_x_learn['ControlOutcome'] = model_c_residual.predict(data.drop(['Treatment', 'Outcome', 'TrueCATE'], axis=1))

    # Prepare the data
    X = data.drop(['Treatment', 'Outcome', 'TrueCATE'], axis=1)
    y = data['Treatment']

    # Fit the logistic regression model
    propensity_model = LogisticRegression()
    propensity_model.fit(X, y)

    # Predict the propensity scores
    data_x_learn['PropensityScore'] = propensity_model.predict_proba(X)[:, 1]

    # calculate the weighted average of the predicted treatment and control outcomes as the predicted outcome
    data_x_learn['PredictedCATE'] = data_x_learn['PropensityScore'] * data_x_learn['TreatmentOutcome'] + (1 - data_x_learn['PropensityScore']) * data_x_learn['ControlOutcome']

    # Calculate the difference between the predicted and true CATE
    data_x_learn['CATEError'] = data_x_learn['PredictedCATE'] - data_x_learn['TrueCATE']

    return data_x_learn


def plot_results(data: pd.DataFrame,
                 sample_frac: float = 0.5):
    """
    plot the results of the CATE estimation
    Args:
        data (pd.DataFrame): contains 'TrueCATE', 'PredictedCATE_t', 'PredictedCATE_s', 'PredictedCATE_x'
    """

    # Sample half of the data
    sampled_data = data.sample(frac=sample_frac, random_state=42)

    # Plot True CATE vs Predicted CATE for each meta learner
    plt.figure(figsize=(10, 6))

    plt.scatter(sampled_data['TrueCATE'], sampled_data['PredictedCATE_t'], label='T-Learner', alpha=0.2, marker='o')
    plt.scatter(sampled_data['TrueCATE'], sampled_data['PredictedCATE_s'], label='S-Learner', alpha=0.2, marker='s')
    plt.scatter(sampled_data['TrueCATE'], sampled_data['PredictedCATE_x'], label='X-Learner', alpha=0.2, marker='^')

    plt.xlabel('True CATE')
    plt.ylabel('Predicted CATE')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()
    plt.grid(True)
    plt.title('True CATE vs Predicted CATE for each meta learner')

    plt.show()


def result_summary(data: pd.DataFrame,
                   data_t_learn: pd.DataFrame,
                   data_s_learn: pd.DataFrame,
                   data_x_learn: pd.DataFrame,
                   plot: bool = False) -> pd.DataFrame:
    """
    Summarize the results of the CATE estimation
    Args:
        data (pd.DataFrame): the original data
        data_t_learn (pd.DataFrame): the data from the T-learner
        data_s_learn (pd.DataFrame): the data from the S-learner
        data_x_learn (pd.DataFrame): the data from the X-learner
    Returns:
        summary (pd.DataFrame): contains 'MSE', 'Bias', 'Variance'
    """

    # combine the results from 3 meta learners
    data['PredictedCATE_t'] = data_t_learn['PredictedCATE']
    data['CATEerror_t'] = data_t_learn['CATEError']
    data['PredictedCATE_s'] = data_s_learn['PredictedCATE']
    data['CATEerror_s'] = data_s_learn['CATEError']
    data['PredictedCATE_x'] = data_x_learn['PredictedCATE']
    data['CATEerror_x'] = data_x_learn['CATEError']

    if plot:
        plot_results(data)

    # calculate MSE, Bias, and Variance for each meta learner
    mse_t = np.mean(data['CATEerror_t'] ** 2)
    mse_s = np.mean(data['CATEerror_s'] ** 2)
    mse_x = np.mean(data['CATEerror_x'] ** 2)

    bias_t = np.mean(data['CATEerror_t'])
    bias_s = np.mean(data['CATEerror_s'])
    bias_x = np.mean(data['CATEerror_x'])

    var_t = np.var(data['CATEerror_t'])
    var_s = np.var(data['CATEerror_s'])
    var_x = np.var(data['CATEerror_x'])

    summary = pd.DataFrame({
        'MSE': [mse_t, mse_s, mse_x],
        'Bias': [bias_t, bias_s, bias_x],
        'Variance': [var_t, var_s, var_x]
    }, index=['T-Learner', 'S-Learner', 'X-Learner'])

    return summary


def main(
        n: int = 5000,
        treatment_proportion: float = 0.1,
        plot: bool = False,
        seed: int = 42
        ) -> pd.DataFrame:
    
    """
    Run the main function to estimate the CATE using T, S, and X-Learner
    and summarize the results
    """

    # Generate synthetic data
    data = generate_synthetic_data(n, treatment_proportion, seed)

    # for the T-learner and X-learner
    # train two models to predict the control and treatment outcomes
    # model for treatment group 
    data_treatment = data[data['Treatment'] == 1]
    X_train_t, X_val_t, y_train_t, y_val_t = split_data(data_treatment, drop_treatment=True)
    results_t = fit_val_model(X_train_t, y_train_t, X_val_t, y_val_t)
    model_t = retrain_best_model(results_t, X_train_t, y_train_t, X_val_t, y_val_t)

    # model for control group
    data_control = data[data['Treatment'] == 0]
    X_train_c, X_val_c, y_train_c, y_val_c = split_data(data_control, drop_treatment=True)
    results_c = fit_val_model(X_train_c, y_train_c, X_val_c, y_val_c)
    model_c = retrain_best_model(results_c, X_train_c, y_train_c, X_val_c, y_val_c)

    # get t-learner results
    data_t_learn = data.copy()
    data_t_learn = t_learner_cate(data_t_learn, model_c, model_t)

    # get x-learner results
    data_x_learn = data.copy()
    data_x_learn = x_learner_cate(data, model_c, model_t)

    # for the S-learner
    # train a single model to predict the outcome
    X_train_s, X_val_s, y_train_s, y_val_s = split_data(data, drop_treatment=False)
    results_s = fit_val_model(X_train_s, y_train_s, X_val_s, y_val_s)
    model_s = retrain_best_model(results_s, X_train_s, y_train_s, X_val_s, y_val_s)

    # get s-learner results
    data_s_learn = data.copy()
    data_s_learn = s_learner_cate(data, model_s)

    # summarize the results
    summary = result_summary(data, data_t_learn, data_s_learn, data_x_learn, plot=plot)

    # add sample size and treatment proportion to the summary
    summary['SampleSize'] = n
    summary['TreatmentProportion'] = treatment_proportion

    return summary