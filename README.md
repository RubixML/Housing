# Housing Price Predictor
This is an example Rubix ML project that predicts house prices using a Gradient Boosted Machine. The dataset was featured in a popular [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) designed to teach advanced regression skills. In this tutorial, you'll learn about regression analysis and the stage-wise additive boosting ensemble [Gradient Boost](https://github.com/RubixML/RubixML#gradient-boost). By the end of the tutorial, you'll be able to submit your own predictions to the competition.

- **Difficulty**: Easy
- **Training time**: Short
- **Memory needed**: < 1G

## Installation
Clone the repository locally:
```sh
$ git clone https://github.com/RubixML/Housing
```
  
Install dependencies:
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial
This tutorial is designed to walk you through a typical regression problem in Rubix using the sale price of a home as the target variable. We are given a training set consisting of 1460 labeled samples and 1459 unknown samples. Each sample contains a heterogeneous mix of categorical and continuous data types. Instead of coverting all of the features to one type, we'll choose the [Gradient Boost](https://github.com/RubixML/RubixML#gradient-boost) regressor because it is capable of handling both data types at once by default.

A Gradient Boosted Machine is a type of *ensemble* estimator that uses [Regression Trees](https://github.com/RubixML/RubixML#regression-tree) to fix up the errors of a *weak* base estimator. It does so in an interative process that involves training a new tree (called a *booster*) on the error residuals of the predictions given by the previous estimator. The *Gradient* in the name comes from the fact that the learner uses Gradient Descent under the hood. The coordination between multiple estimators to act as a single estimator is called *ensemble* learning and Gradient Boost is an example of such an estimator.

### Training
The data are given to us in a CSV file so we'll use the PHP League's [CSV Reader](https://csv.thephpleague.com/) to assist us in extracting the data into a [Labeled](https://github.com/RubixML/RubixML#labeled) dataset object.

> Source code can be found in the [train.php](https://github.com/RubixML/Housing/blob/master/train.php) file in project root.

```php
use Rubix\ML\Datasets\Labeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/train.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
    'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
    'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
    'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
    'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition',
]);

$labels = $reader->fetchColumn('SalePrice');

$dataset = Labeled::fromIterator($samples, $labels);
```

With the dataset object instantiated we can apply the [Numeric String Converter](https://github.com/RubixML/RubixML#numeric-string-converter) to convert all numerical strings to their integer and floating point counterparts. This is necessary since the CSV Reader imports everything as a string by default, including the labels.

> **Note**: Since the Numeric String Converter is not a *Stateful* transformer, it can be applied right away without having to be fitted.

```php
use Rubix\ML\Transformers\NumericStringConverter;

$dataset->apply(new NumericStringConverter());
```

We take care of converting the labels with the `transformLabels()` method on the Labeled dataset object. It takes a function that receives the label and return the transformed label as its only argument. Since the labels of this dataset are the sale prices of each house rounded to the nearest dollar, we'll cast the imported strings (categorical) to integers (continuous) using the built in PHP function `intval()`.

```php
$dataset->transformLabels('intval');
```

We'd like to be able to tell if the model we've trained is any good. As such, we'll need to set some of the data aside for testing purposes. Fortunately, dataset objects make it really easy to randomize and split the dataset for you. Let's pick 80% of the data for training and the remaining 20% to be used for testing.

```php
[$training, $testing] = $dataset->randomize()->split(0.8);
```

The next item on our list is to instantiate the Gradient Boost learner and wrap it in a Persistent Model meta-Estimator so we can save it for later use. 

The first hyper-parameter is the booster instance i.e the tree that is used to fix up the errors of the base estimator. The step size is controlled by the *rate* hyper-parameter. In addition, you can control the maximum number of iterations with the *estimators* parameter and the ratio of training data to feed into each booster with the *ratio* parameter. For a full list of hyper-parameters, see the [API Reference](https://github.com/RubixML/RubixML#gradient-boost).

We choose to use a Regression Tree with max depth of 4 as the boosting estimator, a learning rate of 0.1, and a maximum of 300 iterations using 80% of the training data per iteration.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    new GradientBoost(new RegressionTree(4), 0.1, 100, 0.8),
    new Filesystem(MODEL_FILE, true)
);
```

Now, training is simply a matter of passing in the *training* set to the estimator's `train()` method.

```php
$estimator->train($training);
```

Then we'll dump the training loss at each epoch from the training session so we can visualize it.

```php
$steps = $estimator->steps();
```

Here is an example of what the training loss looks like when its plotted. You can plot the loss yourself by importing the `progress.csv` file into your favorite plotting software. If you're looking for a place to start, we recommend either [Tableu](https://public.tableau.com/en-us/s/) or [Google Sheets](https://www.google.com/sheets/about/).

![MSE Loss](https://raw.githubusercontent.com/RubixML/Housing/master/docs/images/training-loss.svg?sanitize=true)

The remaining data left in the *testing* set is used to generate a [Residual Analysis](https://github.com/RubixML/RubixML#residual-analysis) report which ouputs a number of validation metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), and R Squared (R2). To generate the report we'll need the predictions from the estimator and the labels from the testing set. Lastly, we'll save the report to a JSON file so we can review the performance before saving the model to storage.

```php
use Rubix\ML\CrossValidation\Reports\ResidualAnalysis;

$predictions = $estimator->predict($testing);

$report = new ResidualAnalysis();

$results = $report->generate($predictions, $testing->labels());
```

The results will look something like this.

```json
{
    "mean_absolute_error": 14528.340820915579,
    "median_absolute_error": 10549.487747601735,
    "mean_absolute_percentage_error": 8.911117077298245,
    "mean_squared_error": 397960732.04236144,
    "rms_error": 19948.95315655339,
    "mean_squared_log_error": 0.01637578248729187,
    "r_squared": 0.8986000796796421,
    "error_mean": 44.38514176512214,
    "error_variance": 397958762.0015515,
    "error_skewness": 0.018759888007460087,
    "error_kurtosis": 2.103139242198865,
    "error_min": -80358.30060322193,
    "error_max": 84035.54964507371,
    "cardinality": 292
}
```

Finally, save the model so we can use it later to predict prices of unknown samples.

```php
$estimator->save();
}
```

To run the training script from the project root:
```sh
$ php train.php
```

### Prediction
The model we trained will now be used to generate predictions to submit to the Kaggle competition. We'll load the unknown samples from the `unknown.csv` file into an Unlabeled dataset object and keep track of their *Id* number in a separate array for later.

> Source code can be found in the [predict.php](https://github.com/RubixML/Housing/blob/master/predict.php) file in the project root.

```php
use Rubix\ML\Datasets\Unlabeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/unknown.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
    'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
    'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
    'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
    'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition',
]);

$ids = iterator_to_array($reader->fetchColumn('Id'));

$dataset = Unlabeled::fromIterator($samples);
```

Loading the trained Gradient Boost estimator can be done by passing a [Persister](https://github.com/RubixML/RubixML#persisters) instance pointing to the model in storage to the [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) meta-Estimator.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('housing.model'));
```

Now just call `predict()` with the unknown dataset and write the predictions to a CSV file with their cooresponding Ids. An example of a sample submission can be found in the `sample_submission.csv` file.

```php
$predictions = $estimator->predict($dataset);
```

That's it! Best of luck!

To run the prediction script from the project root:
```sh
$ php predict.php
```

### Wrapup

- Regressors predict *continuous* valued outcomes such as house or stock prices
- Ensemble learning combines the predictions of multiple estimators into one
- The Gradient Boost regressor is an ensemble learner that uses trees to fix the errors of a *weak* base estimator
- Gradient Boost can *by default* handle both categorical and continuous data types
- Data competitions are a great way to practice your data science skills

## Original Dataset
From Kaggle:

> Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
> 
> With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

### References
[1] D. De Cock. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, Volume 19, Number 3.