package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.WeightBias
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution
import org.deeplearning4j.nn.conf.dropout.GaussianNoise
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.weightnoise.WeightNoise
import org.deeplearning4j.nn.weights.WeightInit
import org.junit.Assert.assertEquals
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.RmsProp

val ACTIVATION = Activation.SOFTMAX.activationFunction
val WEIGHT_INIT = WeightInit.ONES
val BIAS_INIT = 1.0
val DIST = TruncatedNormalDistribution(12.0, 1.0)
val DROPOUT = GaussianNoise(192.0)
val WEIGHT_NOISE = WeightNoise(DIST)
val UPDATER = RmsProp()
val BIAS_UPDATER = Adam()
val GRAD_NORM = GradientNormalization.ClipL2PerLayer
val GRAD_NORM_THRESHOLD = 129.0
val L1_WEIGHT = 1.0
val L1_BIAS = 2.0
val L2_WEIGHT = 3.0
val L2_BIAS = 4.0

fun IBaseLayerConf.applyTestConstants() {
    activation = ACTIVATION
    weightInitFunction = WEIGHT_INIT.getWeightInitFunction(DIST)
    biasInit = BIAS_INIT
    dropOut = DROPOUT
    weightNoise = WEIGHT_NOISE
    updater = UPDATER
    biasUpdater = BIAS_UPDATER
    gradientNormalization = GRAD_NORM
    gradientNormalizationThreshold = GRAD_NORM_THRESHOLD
    l1 = WeightBias(L1_WEIGHT, L1_BIAS)
    l2 = WeightBias(L2_WEIGHT, L2_BIAS)
}

fun <T : BaseLayer> T.assertTestConstants() {
    assertEquals(ACTIVATION, activationFn)
//    assertEquals(WEIGHT_INIT, weightInit)
    assertEquals(BIAS_INIT, biasInit, 1e-8)
    assertEquals(DROPOUT, iDropout)
    assertEquals(WEIGHT_NOISE, weightNoise)
    assertEquals(UPDATER, iUpdater)
    assertEquals(BIAS_UPDATER, biasUpdater)
    assertEquals(GRAD_NORM, gradientNormalization)
    assertEquals(GRAD_NORM_THRESHOLD, gradientNormalizationThreshold, 1e-8)
//    assertEquals(L1_WEIGHT, l1, 1e-8)
//    assertEquals(L1_BIAS, l1Bias, 1e-8)
//    assertEquals(L2_WEIGHT, l2, 1e-8)
//    assertEquals(L2_BIAS, l2Bias, 1e-8)
}


