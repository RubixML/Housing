# Housing Price Predictor
This is an example Rubix ML project that predicts house prices using a Gradient Boosted Machine. The dataset was featured in a popular [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) designed to teach advanced regression skills. In this tutorial, you'll learn about regression analysis and the stage-wise additive boosting ensemble [Gradient Boost](https://github.com/RubixML/RubixML#gradient-boost).

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

### Training
The data are given to us in a CSV file so we'll use the PHP League's [CSV Reader](https://csv.thephpleague.com/) to assist us in extracting the data into a [Labeled](https://github.com/RubixML/RubixML#labeled) dataset object.

> **Note**: The full code can be found in the `train.php` file in the root directory.

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

With the dataset object instantiated we can apply the [Numeric String Converter](https://github.com/RubixML/RubixML#numeric-string-converter) to convert all numerical strings to their integer and floating point counterparts. This is necessary since the CSV Reader imports everything as a string by default, including the label.

> **Note**: Since the Numeric String Converter is not a *Stateful* transformer, it can be applied right away without having to be fitted.

```php
use Rubix\ML\Transformers\NumericStringConverter;

$dataset->apply(new NumericStringConverter());
```

We'd like to be able to tell if the model we've trained is any good. As such, we'll need to set some of the data aside for testing purposes. Fortunately, dataset objects make it really easy to randomize and split the dataset for you. Let's pick 80% of the data for training and the remaining 20% to be used for testing.

```php
list($training, $testing) = $dataset->randomize()->split(0.8);
```

The next item on our list is to instantiate the Gradient Boost learner and wrap it in a Persistent Model meta-Estimator so we can save it for later use. A Gradient Boosted Machine is a type of estimator that uses [Regression Trees](https://github.com/RubixML/RubixML#regression-tree) to fix up the errors of a *weak* base estimator. It does so in an interative process that involves training a new tree (called a *booster*) on the error residuals of the predictions given by the previous estimator. The *Gradient* in the name comes from the fact that the learner uses Gradient Descent under the hood. The step size is controlled by the *rate* hyper-parameter. In addition, you can control the maximum number of iterations with the *estimators* parameter and the ratio of training data to feed into each booster with the *ratio* parameter. For a full list of hyper-parameters, see the [API Reference](https://github.com/RubixML/RubixML#gradient-boost).

Since Gradient Boost implements the *Verbose* interface, we set a [Screen Logger](https://github.com/RubixML/RubixML#screen) so we can monitor the training progress in real time.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Other\Loggers\Screen;

$estimator = new GradientBoost(new RegressionTree(4), 0.1, 300, 0.8);

$estimator = new PersistentModel($estimator, new Filesystem(MODEL_FILE));

$estimator->setLogger(new Screen('housing'));
```

Now, training is simply a matter of passing in the *training* set to the estimator's `train()` method. We'll also dump the value of the loss function at each epoch to a CSV file using the League's CSV Writer so we can visualize it later.

```php
use League\Csv\Writer;

$estimator->train($training);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss']);
$writer->insertAll([$estimator->steps()]);
```

Here is an example of what the training loss looks like when plotted. You can plot the loss yourself using your favorite plotting software. If you're looking for a place to start, we recommend either [Tableu](https://public.tableau.com/en-us/s/) or [Google Sheets](https://www.google.com/sheets/about/).

![MSE Loss](https://github.com/RubixML/Housing/blob/master/docs/images/training-loss.png)

The remaining data left in the *testing* set is used to generate a [Residual Analysis](https://github.com/RubixML/RubixML#residual-analysis) report which ouputs a number of validation metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), and R Squared (R2). To generate the report we'll need the predictions from the estimator and the labels from the testing set. Lastly, we'll save the report to a JSON file so we can review the performance before saving the model to storage.

```php
use Rubix\ML\CrossValidation\Reports\ResidualAnalysis;

$predictions = $estimator->predict($testing);

$report = new ResidualAnalysis();

$results = $report->generate($predictions, $testing->labels());

file_put_contents(REPORT_FILE, json_encode($results, JSON_PRETTY_PRINT));
```

The report should look something like this.

```json
{
    "mean_absolute_error": 15007.526455712445,
    "median_absolute_error": 11027.516021091185,
    "mean_squared_error": 478150071.95219266,
    "mean_squared_log_error": 19.985435199240193,
    "rms_error": 21866.642905398,
    "r_squared": 0.9141839883291687,
    "error_mean": -2055.3779961744563,
    "error_variance": 473925493.2450344,
    "error_skewness": 0.248605560779077,
    "error_kurtosis": 7.558982938324787,
    "error_min": -108277.51692212999,
    "error_max": 126464.11156558566,
    "cardinality": 292
}
```

Finally, prompt the user to save the model.

```php
$estimator->prompt();
```

To run the training script from the project root:
```sh
$ php train.php
```

or

```sh
$ composer train
```

### Prediction

On the map ...

## Original Dataset
From Kaggle:

> Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
> 
> With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

### References
[1] D. De Cock. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, Volume 19, Number 3.