<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use League\Csv\Writer;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$dataset = Unlabeled::fromIterator(new CSV('unknown.csv', true))
    ->apply(new NumericStringConverter());

$ids = $dataset->column(0);

$dataset->dropColumn(0);

$estimator = PersistentModel::load(new Filesystem('housing.model'));

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($dataset);

$writer = Writer::createFromPath('predictions.csv', 'w+');

$writer->insertOne(['Id', 'SalePrice']);
$writer->insertAll(array_transpose([$ids, $predictions]));

echo 'Predictions saved to predictions.csv' . PHP_EOL;