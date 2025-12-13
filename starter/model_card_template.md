# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a binary classification model trained to predict whether an individual's income exceeds $50K per year based on U.S. Census data. 

The model used is a Logistic Regression classifier implemented using scikit-learn. Categorical features are one-hot encoded and numerical features are used as-is. The target variable is the `salary` column, which is binarized during training.

The model is trained offline and saved as a serialized artifact for later inference.

## Intended Use

This model is intended for educational purposes as part of a machine learning engineering project. It demonstrates an end-to-end workflow including data preprocessing, model training, evaluation, testing, and deployment.

The model may be used to explore patterns in census data and to illustrate how classification models behave on demographic features. It is not intended for real-world decision-making, automated screening, or any use that impacts individuals directly.


## Training Data

The model is trained on the U.S. Census Income dataset provided in the project starter code. The dataset contains demographic and employment-related features such as age, education, workclass, occupation, race, sex, and hours worked per week.

The training data is preprocessed by removing leading and trailing spaces from categorical values, encoding categorical features using one-hot encoding, and binarizing the target variable (`salary`). The dataset is split into training and test sets using an 80/20 split.


## Evaluation Data
The evaluation data consists of a held-out test split from the same U.S. Census Income dataset used for training. The test set represents approximately 20% of the total dataset and is processed using the same preprocessing steps as the training data.

No external or additional datasets are used for evaluation.


## Metrics
The model is evaluated using precision, recall, and F1-score (FBeta with β=1). These metrics are computed using scikit-learn and provide insight into the model’s classification performance, particularly for imbalanced classes.

In addition to overall performance, the model includes functionality to evaluate performance on slices of the data based on categorical features, allowing inspection of how metrics vary across different demographic groups.

## Ethical Considerations
The dataset used to train this model contains sensitive demographic attributes such as race, sex, and marital status. Models trained on this data may learn and reinforce existing societal biases present in the dataset.

Predictions produced by this model should not be used for decision-making in high-stakes or sensitive contexts. Care should be taken to evaluate model performance across different subgroups to identify potential disparities.

## Caveats and Recommendations
The model is trained on a static snapshot of census data and may not generalize well to populations or conditions outside of the dataset. Performance may degrade if the data distribution changes over time.

Future improvements could include feature scaling, hyperparameter tuning, alternative model architectures, and more comprehensive fairness evaluations. Any deployment of this model should include continuous monitoring and periodic retraining.

