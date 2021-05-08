<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$extractor = new ColumnPicker(new CSV('dataset.csv', true), [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
    'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
    'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
    'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
    'SaleType', 'SaleCondition',
]);

$dataset = Unlabeled::fromIterator($extractor)
    ->apply(new NumericStringConverter());

$estimator = PersistentModel::load(new Filesystem('housing.rbx'));

$logger->info('Making predictions');

$predictions = $estimator->predict($dataset);

$extractor = new ColumnPicker(new CSV('dataset.csv', true), ['Id']);

$ids = array_column(iterator_to_array($extractor), 'Id');

array_unshift($ids, 'Id');
array_unshift($predictions, 'SalePrice');

$extractor = new CSV('predictions.csv');

$extractor->export(array_transpose([$ids, $predictions]));

$logger->info('Predictions saved to predictions.csv');
