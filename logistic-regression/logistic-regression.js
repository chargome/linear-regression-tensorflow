const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 100,
      batchSize: 10,
      decisionBoundary: 0.5,
    }, options);

    this.weights = tf.zeros([this.features.shape[1],1]);
    this.costHistory = []; 
  }

  gradientDescent(features, labels) {    
    const currentGuesses = features.matMul(this.weights).sigmoid();;
    const differences = currentGuesses.sub(labels);
    
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);
    
    this.weights = this.weights
      .sub(slopes.mul(this.options.learningRate))
  }

  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
    for (let i = 0; i < this.options.iterations; i++) {
      for (let x = 0; x < batchQuantity; x++) {
        const startIndex = x * this.options.batchSize;
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [this.options.batchSize, -1],
        );
        const labelSlice = this.labels.slice(
          [startIndex, 0],
          [this.options.batchSize, -1],
        );
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.setCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast('float32');
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    const incorrect = predictions
      .sub(testLabels) // correct guesses are 0 else 1
      .abs() // take absolute value in order to count all faults
      .sum() 
      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);

    if (!this.mean || !this.variance) {
      this.setStandardizationValues(features)
    }

    features = this.standardize(features);
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  setStandardizationValues(features) {
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;
  }

  standardize(features) {
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  setCost() {
    // cross entropy
    const guesses = this.features.matMul(this.weights).sigmoid();

    const termOne = this.labels
      .transpose()
      .matMul(guesses.log());
    
    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
      );
    
    const cost = termOne.add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .get(0, 0);

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;