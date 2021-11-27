package org.deeplearning4j.kotlin.layer

import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise
import org.deeplearning4j.nn.weights.IWeightInit
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.learning.regularization.Regularization

interface IBaseLayerConf {
    var name: String?
    var activation: IActivation
    var weightInitFunction: IWeightInit
    var biasInit: Double
    var dropOut: IDropout?
    var weightNoise: IWeightNoise?
    var updater: IUpdater
    var biasUpdater: IUpdater?
    var gradientNormalization: GradientNormalization
    var gradientNormalizationThreshold: Double
    var l1: WeightBias?
    var l2: WeightBias?
    var regularizations: List<Regularization>
    var regularizationBiases: List<Regularization>
}

data class WeightBias(val weight: Double, val bias: Double)