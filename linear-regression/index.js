require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');



let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
});

const regression = new LinearRegression(
  features,
  labels,
  {
    learningRate: 0.1,
    iterations: 100,
    batchSize: 10,
  },
);

console.log('Training...');
regression.train();
console.log('Training done.');

console.log('Testing...');
const r2 = regression.test(testFeatures, testLabels);
console.log('Testing done. R2: ', r2);

console.log('Predicting...');

const prediction = regression.predict([
  ['661', '1.4', '238'], // ferrari 488 GTB
  ['40', '0.9', '100'],
  ['300', '1.4', '400'],
]);

console.log('Prediction:');
prediction.print();

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'MSE', 
});

