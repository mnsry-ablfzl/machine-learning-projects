# GridSearchCV Report

## Objective

This report summarizes the `Untitled.ipynb` notebook in the `gridsearchcv` project. The notebook demonstrates how to tune machine learning hyperparameters using cross-validation, `GridSearchCV`, and `RandomizedSearchCV` on the Iris dataset.

## Dataset

- Dataset: Iris flower classification dataset from scikit-learn.
- Samples: 150
- Features: 4
- Classes: setosa, versicolor, virginica

Example rows used in the notebook:

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | flower |
| --- | --- | --- | --- | --- |
| 4.6000 | 3.2000 | 1.4000 | 0.2000 | setosa |
| 5.3000 | 3.7000 | 1.5000 | 0.2000 | setosa |
| 5.0000 | 3.3000 | 1.4000 | 0.2000 | setosa |
| 7.0000 | 3.2000 | 4.7000 | 1.4000 | versicolor |
| 6.4000 | 3.2000 | 4.5000 | 1.5000 | versicolor |

## Baseline Model

The notebook first trains an SVM classifier with an RBF kernel, `C=30`, and `gamma='auto'` using a 70/30 train-test split.

- Test split accuracy: 1.0000

## Cross-Validation Checks

The notebook compares several SVM configurations using 5-fold cross-validation.

| kernel | C | fold_scores | mean_score |
| --- | --- | --- | --- |
| linear | 10 | 1.000, 1.000, 0.900, 0.967, 1.000 | 0.9733 |
| rbf | 10 | 0.967, 1.000, 0.967, 0.967, 1.000 | 0.9800 |
| rbf | 20 | 0.967, 1.000, 0.900, 0.967, 1.000 | 0.9667 |

## Manual Hyperparameter Search

Before using `GridSearchCV`, the notebook manually loops over SVM kernels and `C` values and averages 5-fold cross-validation scores.

| kernel | C | mean_score |
| --- | --- | --- |
| rbf | 1 | 0.9800 |
| rbf | 10 | 0.9800 |
| linear | 1 | 0.9800 |
| linear | 10 | 0.9733 |
| rbf | 20 | 0.9667 |
| linear | 20 | 0.9667 |

## GridSearchCV Results

`GridSearchCV` evaluates all combinations of `C` and `kernel` for the SVM classifier.

| param_C | param_kernel | mean_test_score | rank_test_score |
| --- | --- | --- | --- |
| 1 | linear | 0.9800 | 1 |
| 1 | rbf | 0.9800 | 1 |
| 10 | rbf | 0.9800 | 1 |
| 10 | linear | 0.9733 | 4 |
| 20 | rbf | 0.9667 | 5 |
| 20 | linear | 0.9667 | 6 |

Best `GridSearchCV` result:

- Best score: 0.9800
- Best parameters: `{'C': 1, 'kernel': 'rbf'}`

## RandomizedSearchCV Results

`RandomizedSearchCV` samples two parameter combinations from the same SVM search space.

| param_C | param_kernel | mean_test_score | rank_test_score |
| --- | --- | --- | --- |
| 1 | rbf | 0.9800 | 1 |
| 1 | linear | 0.9800 | 1 |

## Model Comparison

The notebook also compares SVM, Random Forest, and Logistic Regression using `GridSearchCV` for each model family.

| model | best_score | best_params |
| --- | --- | --- |
| svm | 0.9800 | {'C': 1, 'kernel': 'rbf'} |
| logistic_regression | 0.9800 | {'C': 5} |
| random_forest | 0.9667 | {'n_estimators': 10} |

## Conclusion

- Grid search provides a systematic way to evaluate every configured hyperparameter combination.
- For this Iris dataset, SVM configurations with `C=1` tied for the strongest cross-validation score, with `GridSearchCV` selecting the RBF kernel.
- Randomized search is useful when the search space is large, but with only two sampled configurations it may miss the best available option.
- Cross-validation gives a more reliable estimate than relying on one train-test split.
