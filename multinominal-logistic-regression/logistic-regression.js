const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    this.options = Object.assign({
      learningRate: 0.1,
      iterations:  100,
      batchSize: 10,
      decisionBoundary: 0.5,
    }, options);

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    this.costHistory = [];
  }

  gradientDescent(features, labels) {    
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);
    
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);
    
    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
    for (let i = 0; i < this.options.iterations; i++) {
      for (let x = 0; x < batchQuantity; x++) {
        this.weights = tf.tidy(() => {
          const startIndex = x * this.options.batchSize;
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [this.options.batchSize, -1],
          );
          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [this.options.batchSize, -1],
          );
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      this.setCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions
      .notEqual(testLabels) // correct guesses are 0 else 1
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

    const filler = variance.cast('bool').logicalNot().cast('float32');

    this.mean = mean;
    this.variance = variance.add(filler);
  }

  standardize(features) {
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  setCost() {
    // cross entropy
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid();

      const termOne = this.labels
        .transpose()
        .matMul(
          guesses
            .add(1e-7) // add tiny value in order to never execute log(0)
            .log()
        );
      
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(
          guesses
            .mul(-1)
            .add(1)
            .add(1e-7) // add tiny value in order to never execute log(0)
            .log()
        );
      
      return termOne.add(termTwo)
          .div(this.features.shape[0])
          .mul(-1)
          .get(0, 0);
    });
    
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