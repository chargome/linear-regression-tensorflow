require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');


let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  dataColumns: ['horsepower', 'displacement', 'weight',],
  labelColumns: ['passedemissions'],
  shuffle: true,
  splitTest: 50,
  converters: {
    passedemissions: value => (value === 'TRUE' ? 1 : 0),
  }
});

const regression = new LogisticRegression(
  features,
  labels,
  {
    learningRate: 0.5,
    iteration: 100,
    batchSize: 10,
    decisionBoundary: 0.5,
  },
);

regression.train();

console.log(regression.test(testFeatures, testLabels));

const prediction = regression.predict([
  ['130', '307', '1.75'],
  ['95', '113', '1.19'],
  ['200', '203', '1.00'],
]);

prediction.print();

plot({
  x: regression.costHistory.reverse(),
  xLabel: '# Iterations',
  yLabel: 'Cost'
});