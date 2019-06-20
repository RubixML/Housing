<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Persisters\Filesystem;
use League\Csv\Reader;
use League\Csv\Writer;

const MODEL_FILE = 'housing.model';
const PREDICTIONS_FILE = 'predictions.csv';

ini_set('memory_limit', '-1');

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ House Prices Predictor using Gradient Boosted Machine         ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

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

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$predictions = $estimator->predict($dataset);

$writer = Writer::createFromPath(PREDICTIONS_FILE, 'w+');
$writer->insertOne(['Id', 'SalePrice']);
$writer->insertAll(array_map(null, $ids, $predictions));

echo 'Predictions saved to ' . PREDICTIONS_FILE . PHP_EOL;