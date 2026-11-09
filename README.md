# Income prediction with basic machine learning models

This project compares classification models on the Adult Census dataset. The task is to predict whether a record belongs to the **`>50K` annual income class**. I built it to practice data cleaning, exploratory analysis, preprocessing, cross-validation, model comparison, and error interpretation.

The complete analysis, charts, code, and saved results are in [adult-income-classification.ipynb](adult-income-classification.ipynb).

## Dataset

The project uses the [Adult dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). The supplied files are `adult/adult.data` for development and `adult/adult.test` for final evaluation. The source notes are in `adult/adult.names`.

The data has 14 original predictors and one income label. The notebook removes `education_num` because it repeats the information in `education`, leaving 13 predictors. Unknown categories marked `?` are handled inside the model pipelines. It removes duplicate rows and 28 rows whose predictor values also appear in the test file **from development only**. The final analysis uses **32,511 development rows** and the supplied **16,281 test rows**.

## Workflow

1. **Load and audit:** Read both files, normalize the test labels, inspect data types and missing values, and check duplicates and shared predictor rows.
2. **Explore:** Plot the target balance, unknown values, numeric distributions, and income rates for common categories using development data.
3. **Prepare features:** Impute missing values and one-hot encode categories inside each model pipeline. Scale numeric values for Logistic Regression.
4. **Compare models:** Use the same three stratified cross-validation folds for a majority-class baseline, Logistic Regression, Decision Tree, Bagging, Random Forest, Gradient Boosting, and XGBoost.
5. **Select and evaluate:** Select by mean cross-validation F1 for the `>50K` class, fit on all development rows, and compare results on the supplied test file.
6. **Interpret:** Review precision, recall, F1, ROC-AUC, a confusion matrix, example errors, subgroup error rates, and feature importance.

## Results

These are the saved results from the notebook. All values below are rounded to three decimals.

| Model | CV F1 | Test accuracy | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.000 | 0.764 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.658 | 0.851 | 0.728 | 0.586 | 0.649 |
| Decision Tree | 0.660 | 0.859 | 0.772 | 0.572 | 0.657 |
| Bagging | 0.662 | 0.863 | 0.789 | 0.576 | 0.666 |
| **Random Forest** | **0.682** | **0.865** | 0.772 | **0.611** | **0.682** |
| Gradient Boosting | 0.626 | 0.855 | 0.811 | 0.504 | 0.622 |
| XGBoost | 0.661 | **0.865** | 0.809 | 0.563 | 0.664 |

Random Forest had the highest development cross-validation F1 and was selected before examining the test results. It found **2,348 of 3,846** `>50K` test records, missed **1,498**, and produced **695 false positives**. The majority baseline reached 76.4% accuracy by always predicting `<=50K`, but its F1 for `>50K` was zero. This is why the project compares more than accuracy.

## Run the notebook

From the project root, install the notebook tools and libraries:

```bash
python -m pip install notebook ipykernel pandas numpy matplotlib scikit-learn xgboost
jupyter notebook adult-income-classification.ipynb
```

Run the cells from top to bottom. XGBoost is required for the model comparison. The saved run used Python **3.11.11**, pandas **2.2.3**, NumPy **1.26.4**, scikit-learn **1.6.1**, matplotlib **3.10.1**, and XGBoost **3.2.0**. All data paths are relative to the project root.

## Limitations and attribution

The Adult data comes from historical census records and includes sensitive attributes. The models and subgroup checks are for learning; their associations do not explain causes or support employment or pay decisions.

Dataset citation: Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5XW20](https://doi.org/10.24432/C5XW20). UCI lists the dataset under [CC BY 4.0](https://archive.ics.uci.edu/dataset/2/adult).
