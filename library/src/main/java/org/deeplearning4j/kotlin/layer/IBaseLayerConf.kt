package org.deeplearning4j.kotlin.layer

import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.learning.config.IUpdater

interface IBaseLayerConf {
	var activation: IActivation
	var weightInit: WeightInit
	var biasInit: Double
	var dist: Distribution?
	var dropOut: IDropout?
	var weightNoise: IWeightNoise?
	var updater: IUpdater
	var biasUpdater: IUpdater?
	var gradientNormalization: GradientNormalization
	var gradientNormalizationThreshold: Double
	var l1Weights: Double
	var l1Bias: Double
	var l2Weights: Double
	var l2Bias: Double
}