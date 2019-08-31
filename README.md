# Housing Price Predictor
An example Rubix ML project that predicts house prices using a Gradient Boosted Machine (GBM) and a popular dataset from a [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). In this tutorial, you'll learn about regression and the stage-wise additive boosting ensemble called [Gradient Boost](https://docs.rubixml.com/en/latest/regressors/gradient-boost.html). By the end of the tutorial, you'll be able to submit your own predictions to the Kaggle competition.

- **Difficulty**: Easy
- **Training time**: < 5 Minutes
- **Memory needed**: < 1G

## Installation
Clone the repository locally using [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/Credit
```
  
Install dependencies using [Composer](https://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial

### Introduction
[Kaggle](https://www.kaggle.com) is a platform that allows you to test your data science skills by engaging with contests. This tutorial is designed to walk you through a regression problem in Rubix ML using the Kaggle housing prices challenge as an example. We are given a training set consisting of 1460 labeled samples that we'll use to train the learner and 1459 unlabeled samples for making predictions. Each sample contains a heterogeneous mix of categorical and continuous data types. Our goal is to build an estimator that correctly predicts the sale price of a house. We'll choose [Gradient Boost](https://docs.rubixml.com/en/latest/regressors/gradient-boost.html) as our estimator since it is capable of handling both data categorical and continuous data types at once.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/Housing/blob/master/train.php) file in project root.

### Extracting the Data
The data are given to us in two separate CSV files - `dataset.csv` which has labels for training and `unknown.csv` for predicting. Each feature column is denoted by a title in the CSV header which we'll use to identify the column. The PHP League's [CSV Reader](https://csv.thephpleague.com/) will assist us in extracting the data from file.

```php
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
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
```

### Dataset Preparation
The `getRecords()` and `fetchColumn()` methods on the Reader instance return iterators which we'll now load into a Labeled dataset object using the `fromIterator()` static factory method.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($samples, $labels);
```

Next we'll apply a series of transformations to the training set to prepare it for the learner. By default, the CSV Reader imports everything as a string type - therefore, we must convert the numerical values to integers and floating point numbers beforehand so they can be recognized by the learner as continuous features. The [Numeric String Converter](https://docs.rubixml.com/en/latest/transformers/numeric-string-converter.html) will handle this for us. Since some feature columns contain missing data, we'll also apply the [Missing Data Imputer](https://docs.rubixml.com/en/latest/transformers/missing-data-imputer.html) which replaces unknown values (denoted by a "?") with a pretty good guess. Lastly, since the labels are also meant to be continuous, we'll apply a separate transformation to the labels of the trainig set using a standard PHP function `intval()` which converts values to integers.

```php
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MissingDataImputer;

$dataset->apply(new NumericStringConverter())
    ->apply(new MissingDataImputer('?'))
    ->transformLabels('intval');
```

### Instantiating the Learner
A Gradient Boosted Machine (GBM) is a type of ensemble estimator that uses [Regression Trees](https://docs.rubixml.com/en/latest/regressors/regression-tree.html) to fix up the errors of a *weak* base learner. It does so in an iterative process that involves training a new Regression Tree (called a *booster*) on the error residuals of the predictions given by the previous estimator. Thus, GBM produces an additive model whose predictions become more refined as the number of boosters are added. The coordination of multiple estimators to act as a single estimator is called *ensemble* learning.

Next we'll create the estimator instance by instantiating [Gradient Boost](https://docs.rubixml.com/en/latest/regressors/gradient-boost.html) and wrapping it in a [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) meta-estimator so we can save it to make predictions later in another process.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    new GradientBoost(new RegressionTree(4), 0.1),
    new Filesystem('housing.model', true)
);
```

The first two hyper-parameters of Gradient Boost are the booster's settings and the learning rate respectively. For this example, we'll use a standard Regression Tree with a maximum depth of 4 as the booster with a learning rate of 0.1 but feel free to play with these settings on your own.

The Persistent Model meta-estimator takes the GBM instance as its first parameter and a Persister object as the second. The [Filesystem](https://docs.rubixml.com/en/latest/persisters/filesystem.html) persister is responsible for storing and loading the model on disk and takes the path of the model file as a parameter. In addition, we'll tell the persister to keep a copy of every saved model by turning history mode on.

### Setting a Logger
Since both Persistent Model and Gradient Boost implement the [Verbose](https://docs.rubixml.com/en/latest/verbose.html) interface, we can monitor progress during training by setting a logger instance on the learner. The built-in [Screen](https://docs.rubixml.com/en/latest/other/loggers/screen.html) logger will suffice for this example, but you can use any PSR-3 compatible logger.

```php
$estimator->setLogger(new Screen('housing'));
```

### Training
Now we're ready to train the learner by calling the `train()` method with the training dataset.

```php
$estimator->train($dataset);
```

### Validation Score and Loss
During training, the learner will record the validation score and the training loss at each epoch. The validation score is calculated using the default [R Squared](https://docs.rubixml.com/en/latest/cross-validation/metrics/r-squared.html) metric on a hold out portion of the training set. Contrariwise, the training loss is the value of the cost function (in this case the L2 loss) computed over the training data. We can vizualize the training progress by plotting these metrics. To export the scores and losses you can call the additional `scores()` and `steps()` methods respectively.

```php
$scores = $estimator->scores();

$losses = $estimator->steps();
```

Here is an example of what the validation score and training loss looks like when they are plotted. The validation score should be getting better with each epoch as the loss decreases. You can generate your own plots by importing the `progress.csv` file into your favorite plotting software.

![R Squared Score](https://raw.githubusercontent.com/RubixML/Housing/master/docs/images/validation-score.svg?sanitize=true)

![L2 Loss](https://raw.githubusercontent.com/RubixML/Housing/master/docs/images/training-loss.svg?sanitize=true)


### Saving
Lastly, save the model so it can be used later to predict the house prices of the unknown samples.

```php
$estimator->save();
```

### Inference
The goal of the Kaggle contest is to predict the correct sale prices of each house given a list of unknown samples. If all went well during training, we should be able to achieve good results with just this basic example. We'll start by importing the unknown samples from the `unknown.csv` file.

> **Note:** The source code for this example can be found in the [predict.php](https://github.com/RubixML/Housing/blob/master/predict.php) file in the project root.

```php
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
```

Notice that we are also importing the ID numbers from the ID column into a separate array. We'll need these numbers to submit to the contest later.

Since the samples in `unknown.csv` are unlabeled, we'll need to instantiate an [Unlabeled](https://docs.rubixml.com/en/latest/datasets/unlabeled.html) dataset object this time.

```php
use Rubix\ML\Datasets\Unlabeled;

$dataset = Unlabeled::fromIterator($samples);
```

### Load Model from Storage
Now let's load the persisted Gradient Boost model from storage using the static `load()` method on Persitent Model. Loading can be done by passing a [Persister](https://docs.rubixml.com/en/latest/persisters/api.html) instance pointing to the model in storage.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('housing.model'));
```

### Make Predictions
To obtain the predictions from the model, simply call `predict()` with the dataset. Then submit those predictions along with their IDs to the [contest page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

```php
$predictions = $estimator->predict($dataset);
```

That's it! Best of luck on the competition!

### Wrapup

- Regressors are a type of estimator that predict continuous valued outcomes such as house prices
- Ensemble learning combines the predictions of multiple estimators into one
- The Gradient Boost regressor is an ensemble learner that uses Regression Trees to fix the errors of a *weak* base estimator
- Gradient Boost can handle both categorical and continuous data types by default
- Data science competitions are a great way to practice your skills

### Next Steps
Have a look at the Gradient Boost [documentation page](https://docs.rubixml.com/en/latest/regressors/gradient-boost.html) for a full description of the hyper-parameters. Then, try tuning those parameters for better results. Another way to achieve better results is by filtering noise samples or outliers from the dataset. For example, you may want to filter out extremely large and expensive houses from the training set.

## Original Dataset
From Kaggle:

> Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
> 
> With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

### References
[1] D. De Cock. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, Volume 19, Number 3.
