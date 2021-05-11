---
title: HSE University - Week 2 - Exploratory Data Analysis
author: ''
date: '2021-02-22'
categories: [course, competitive programming, python]
tags: [course, competitive programming, python]
summary: 'Personal notes of `HSE University - How to Win a Data Science Competition`, Lecture 2: Exploratory Data Analysis'
reading_time: yes
image:
  caption: ''
  focal_point: ''
  preview_only: true
type: #today-i-learned
draft: true
---

# <span style="color:#FF9F1D"> Approaching a competition</span>

First steps:

1. **Understand the task and the data**, build an intuition about the columns, what kind of features are important, and so forth.

2. **Don't start modeling**. Use descriptive visualization to understand better the features.

3. **Visualize patterns in the data**. Ask yourself why is the data is constructed the way it is and try to build an initial hypothesis to check.

# <span style="color:#FF9F1D"> Building intuition about the data </span>

Steps to build a sense of the dataset:

**1. Get domain knowledge**

Most of the time, you start a competition without knowing anything about the topic. It's normal to use **Google and Wikipedia**. Read several **articles** about the topic to understand the description of the columns, and make sense of the data.

**2. Check if the data correspond to the domain knowledge**

**Check the values in the columns**. Do they make sense according to the data description, or your knowledge of the topic?

Mistakes can create advantages in competitions. Create a new feature (column "is_correct") simply with a boolean indicating if the row makes sense (True) or not (False)

**3. Understand how the data was generated**

**Do the train set and validation set come from the same distribution?**

If not, modeling on the training set will never approximate the global minimum. Improving the model in the validation set does not without improving the better public score in the leaderboard can be a symptom of different data generation processes.

# <span style="color:#FF9F1D"> Exploring anonymized data </span>

Sometimes the organizers hash (normalize or code) some sensible data, or leak some hint about the data.

Try:

- Print the unique values: *unique()*
- Check the mean and standard deviation. If it's close to 0 and 1, it was normalized.
- If normalized, sort the unique values of the column ( *np.sort(X_train.column_1.unique()*) and divide the column by the constant difference that you observe.

If you cannot investigate the exact numbers behind the hash (decode), at least guess the data type (numerical, categorical, ordinal...) to process the columns accordingly to the type.

Decoding or guessing the hash values is important for choosing the rigth model and improving model performance.

# <span style="color:#FF9F1D"> Visualizations </span>

**1. Exploration of individual features**

Histograms (*plt.hist(x)*): Check the number of bins and scale to get the distribution picture are right. An **abnormal peak in the distribution could signal that the missing values were labeled as that number by organizers.**

Example:

![](./images/L2_hist.png "Example of an abnormal pick in a distribution")

Then, the value can be replaced for NaN, by -999, or simply including a boolean column indicating that in that row it was this abnormal value.

**General statistics** (*df.describe()*) can also signal not normally distributed values.

**2. Exploration of feature relations.**

**Scatterplots** (*plt.scatter(x1, x2)* and *pd.scatter_matrix(df)*). We can use scatterplots to check if the data distribution of the train and test data are the same:

![](./images/L2_scatter.png " ")


**Correlation matrix** (*df.corr()*) and **clustering plots** (*df.mean().sort_values().plot(style = ".")*) can help to detect relationship between the variables.

# <span style="color:#FF9F1D"> Dataset cleaning and other things to check </span>

- If a feature is constant in the train set but not in the test set or vice versa, it is better to remove it.

In general, constants does not help as in doesn't signal any difference in the characteristics between labels. In pandas you can find constants by:

*df.nunique(drop = True).sort_values().head()*

If the feature has 1 unique value, is a constant. To drop them

*constants = df.nunique(drop = True)*

*constant_columns = constants.loc[constants == 1].index.tolist()*
*df.drop(columns = constant_columns, inplace = True)*

- If two columns are duplicated (or scaled), remove them by:

*df.T.drop_duplicates()*

- Explore the name of the columns. If the name is something like *RP_17, RP18, RP_19, VAR_89, VAR_90*, it could indicate a sequence or a time series.

- If two rows are duplicated (or scaled), understand why they are duplicated. It is a mistake by the organizers?

- Check if the data was shuffled.

The index may indicate that the train and test data are indeed from the same dataset: train data frame indexes being 1,3,5,7 and test data indexes 2,4,6,8 for example.

![](./images/L2_correlation_sorted.png " ")

# <span style="color:#FF9F1D"> Validation strategies </span>

In an usual fashion, data is splitted into train, test and validation chunks:

![](./images/L2_data_division.png " ")

In competitions, the test data is divided into public and private. Public split is used to calculate your score into the public leaderboard and as a measure of how your model performs. However, the final score is determined by how the model performs in the unseen private split that only the organizers have access to.

![](./images/L2_data_division_comp.png " ")

Avoid overfitting the public split. A high public leaderboard can be improved just by trying thousands of times. If the model overfits the public test split, is more likely that performs poorly on the private split.


### Holdout split

*sklearn.model_selection.ShuffleSplit*

The simplest split. It splits the data into train data and validation data, without overlapping. Train and validation observations cannot overlap. Otherwise, the model would overfit. It would learn the specific parameters that it has seen for the observation in the training set that fits perfectly the validation set, not because there are optimal parameters but because is the same observation.

![](./images/L2_holdout.png " ")

### K-fold Cross-Validation

*sklearn.model_selection.KFold*

K-fold is holdout done multiple times. Recommended for large enough data. It splits the data into different parts and iterates through them, using every part as a validation set only once.

![](./images/L2_kfold.png " ")


### Leave-One-out Cross-Validation or Jackknife

*sklearn.model_selection.LeaveOneOut*

All *n* data points are repeatedly split into a training set containing all but one observation, and the validation set contains only that observation.

- The first split leaves out observation number 1 for validation and calculates MSE, or another performance metric.

- The second split leaves out observation number 2, and calculates MSE. Repeat for all *n* observations.

![](./images/L2_leave.png " ")


All the validation strategies split the dataset into chunks. The main difference is how large are these chunks.

- Only one iteration and one chunk: Holdout.
- K iteration and chunks: K-fold CV.
- All the iterations and chunks possible without repetition: Jackknife.


## Stratification

In small samples, the distribution of the different classes or labels is an important aspect to take into account.

As an illustrative example, we have the following classification problem: separating between red dots and blue dots for an 8 observation dataset. Using any strategy, there is a high probability that the train split can group all the red dots, and the validation split all the blue dots. The model cannot learn how to classify between them, as it is only trained with blue dots.

Generalization: if there are not enough observations between classes in the training splits, the model cannot learn to differentiate the target.

Stratification makes sure that the distribution of classes is the same in the train set and the validation set. Stratification is useful (or necessary) for:

- Small datasets.
- Unbalanced datasets (over-represented label).
- Multiclass classification.

It never hurts to stratify. In large datasets, the sample target distribution will be the same as the population target distribution because of the law of the large numbers. The bigger the sample, the more similar is to the total population.

# <span style="color:#FF9F1D"> Data splitting strategies </span>

The way the split is made can change the performance of the model significantly. For example, in time series, the train-validation split can be made using:

- Previous observations as train data and present observations as validation.
- Both past and present observations to train the data, and using observations in between as validation data.

**The features we generate depend on the train-test data splitting method.**

![](./images/L2_splits_timeseries.png " ")

**When it makes sense to use sequential or timewise split?**

When choosing between a random or a sequential split, it must be taken into account the structure of the data itself and the covariates that can be created. If the test data is in the future time sequence, it makes more sense to split in a sequential way or timewise.

If we generate features that describe time-based patterns, a random-based split will not get a reliable validation.

**When it makes sense to use random split?**

When the observations are independent.

As a general rule: **set up the validation split to mimic the train/test split of the competition**.

# <span style="color:#FF9F1D"> Problems occurring during validation </span>

Problem 1: Validation stage. Getting optimal parameters for different folds. Causes:

- Too little data.
- Too diverse and inconsistent data.

Check:

1. Average scores from different KFold splits.
2. Tune model on one split, evaluate the score on the other.

Problem 2: Submission stage. Scores in the validation do not match the test set (Leaderboard).

- It is usually because the validation split is different from the train/test split.

Check:

1. There is too little data in the test set.
2. Overfit?
3. Splitting strategy
4. The distribution of the target of sample data and test data. They might be different?


# <span style="color:#FF9F1D"> How to Select Your Final Models in a Kaggle Competition </span>

1. Always do cross-validation to get a reliable metric. Keep in mind the CV score can be optimistic, and your model could be still overfitting.

2. Trust your CV score, and not your leaderboard score. The leaderboard score is scored only on a small percentage of the full test set.

3. For the final 2 models, pick very different models. Picking two very similar solutions means that your solutions either fail together or win together, effectively meaning that you only pick


# <span style="color:#FF9F1D"> Data Leakage </span>

Leakage is a piece of unexpected information in the data that allows us to make unrealistically good predictions.

The model will be able to find the labels way easier using leakages instead of true features. In other words, machine learning algorithms will focus on actually useless features. The features only act as proxies for the leakage indicator.

For example, the task is dividing ads between sponsored or not sponsored. However, all the sponsored ads come after the last non-spponsored ones. Then, it doesn't matter how many or how good are the features, with finding the timestamp of the ads is enough for a classifier to classify the ads.

- Time series

When you enter a time serious competition at first, check train, public, and private splits. **If even one of them is not on time, then you found a data leak.**

- Images

We often have more than just train and test files. For example, a lot of images or text in the archive. In such a case, we can't access some meta information, file creation date, image resolution etcetera. It turns out that this meta-information may be connected to the target variable.

- Identifiers

IDs are unique identifiers of every row usually used for convenience. It makes no sense to include them in the model. It is assumed that they are automatically generated.

In reality, that's not always true. ID may be a hash of something, probably not intended for disclosure. It may contain traces of information connected to the target variable.

**Adding the ID as a feature slightly improves the result.**

- Row order

In a trivial case, data may be shuffled by the target variable.

If there is some kind of row duplication, rows next to each other usually have the same label.

# <span style="color:#FF9F1D"> Leaderboard probing </span>

Sometimes, it's possible to submit predictions in such a way that will give out information about private data.


More [here](https://www.kaggle.com/dansbecker/data-leakage)
