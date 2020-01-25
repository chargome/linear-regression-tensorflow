require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
const plot = require('node-remote-plot');
const LogisticRegression = require('./logistic-regression');
const mnist = require('mnist-data');


const loadData = () => {
  const mnistData = mnist.training(0, 60000);

  const features = mnistData.images.values.map(value => ( _.flatMap(value) ));
  const encodedLabels = mnistData.labels.values.map(value => {
    const row = new Array(10).fill(0);
    row[value] = 1;
    return row;
  });

  return { features, labels: encodedLabels };
}

const createRegressionObject = () => {
  const { features, labels } = loadData();
  return new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 80,
    batchSize: 500,
  });
}

const regression = createRegressionObject();

regression.train();

 // testing
 const mnistTestData = mnist.testing(0, 10000);
 const testFeatures = mnistTestData.images.values.map(value => ( _.flatMap(value) ));
 const testEncodedLabels = mnistTestData.labels.values.map(value => {
  const row = new Array(10).fill(0);
  row[value] = 1;
  return row;
});

const accurracy = regression.test(testFeatures, testEncodedLabels);

console.log('Accurracy: ', accurracy);

plot({
  x: regression.costHistory.reverse(),
})
