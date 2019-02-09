<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\ResidualAnalysis;
use League\Csv\Reader;
use League\Csv\Writer;

const MODEL_FILE = 'housing.model';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Housing Price Predictor using a Gradient Boosted Machine      ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$reader = Reader::createFromPath(__DIR__ . '/training.csv')
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

$dataset->apply(new NumericStringConverter());

$dataset->transformLabels(function ($label) {
    return (int) $label;
});

$estimator = new PersistentModel(
    new GradientBoost(new RegressionTree(3), 0.1, 300, 0.3),
    new Filesystem(MODEL_FILE)
);

$estimator->setLogger(new Screen('housing'));

list($training, $testing) = $dataset->randomize()->split(0.8);

$estimator->train($training);

$writer = Writer::createFromPath('progress.csv', 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_map(null, $estimator->steps(), []));

$predictions = $estimator->predict($testing);

$report = new ResidualAnalysis();

$results = $report->generate($predictions, $testing->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();
}