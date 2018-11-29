<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;
use League\Csv\Writer;

const OUTPUT_FILE = 'tsne.csv';

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ House Prices Dataset Visualizer using t-SNE                   ║' . "\n";
echo '║                                                               ║' . "\n";
echo '╚═══════════════════════════════════════════════════════════════╝' . "\n";
echo "\n";

$reader = Reader::createFromPath(__DIR__ . '/train.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$header = [
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
];

$samples = $reader->getRecords($header);
$labels = $reader->fetchColumn('SalePrice');

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(200);

$converter = new NumericStringConverter();
$encoder = new OneHotEncoder();

$dataset->apply($converter);

$encoder->fit($dataset);

$dataset->apply($encoder);

$embedder = new TSNE(2, 30, 12., 1000, 1.0, 0.3, 1e-6, new Minkowski(3.));

echo 'Embedding started ...';

$start = microtime(true);

$samples = $embedder->embed($dataset);

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

$dataset = Labeled::quick($samples, $dataset->labels());

$writer = Writer::createFromPath(OUTPUT_FILE, 'w+');
$writer->insertOne(['x', 'y', 'label']);
$writer->insertAll($dataset->zip());

echo 'Embedding saved to ' . OUTPUT_FILE . '.' . "\n";