<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Unlabeled::fromIterator(new CSV('unknown.csv', true))
    ->apply(new NumericStringConverter());

$ids = $dataset->column(0);

$dataset->dropColumn(0);

$estimator = PersistentModel::load(new Filesystem('housing.model'));

$logger->info('Making predictions');

$predictions = $estimator->predict($dataset);

Unlabeled::build(array_transpose([$ids, $predictions]))
    ->toCSV(['Id', 'SalePrice'])
    ->write('predictions.csv');

$logger->info('Predictions saved to predictions.csv');