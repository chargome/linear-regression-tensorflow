const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 100,
      batchSize: 10,
    }, options);

    this.weights = tf.zeros([this.features.shape[1],1]);
    this.mseHistory = [];
  }

  gradientDescent(features, labels) {
    /*
    ---------- LODASH VERSION OF GD: ----------
    const currentGuessesForMPG = this.features.map(row => {
      return this.m * row[0] + this.b;
    });

    const bSLope = _.sum(currentGuessesForMPG.map((guess, i) => {
      return guess - this.labels[i][0]
    })) * 2 / this.features.length;

    const mSLope = _.sum(currentGuessesForMPG.map((guess, i) => {
      return -1 * this.features[i][0] * (this.labels[i][0] - guess);
    })) * 2 / this.features.length;

    this.b = this.b - bSLope * this.options.learningRate;
    this.m = this.m - mSLope * this.options.learningRate;
    console.log(this.m, this.b);
    */
    
    const currentGuesses = features.matMul(this.weights);
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
      this.setMSE();
      this.updateLearningRate();
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);
    
    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .get();
    
    const tot = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .get();

    return 1 - res / tot;
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
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

  setMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;