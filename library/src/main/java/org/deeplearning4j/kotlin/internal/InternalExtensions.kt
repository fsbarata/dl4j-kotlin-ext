package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.IFeedForwardLayerConf
import org.deeplearning4j.kotlin.layer.ILossLayerConf
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer


internal fun <T : BaseOutputLayer.Builder<T>> ILossLayerConf.applyTo(builder: T) {
    builder.lossFunction(lossFunction)
}

internal fun <T : FeedForwardLayer.Builder<T>> IFeedForwardLayerConf.applyTo(builder: T) {
    builder.nIn(nIn)
    builder.nOut(nOut)
}

internal fun <T : BaseLayer.Builder<T>> IBaseLayerConf.applyTo(builder: T) {
    builder.activation(activation)
    builder.weightInit(weightInitFunction)
    builder.biasInit(biasInit)
    builder.dropOut(dropOut)
    builder.weightNoise(weightNoise)
    builder.updater(updater)
    builder.biasUpdater(biasUpdater)
    builder.gradientNormalization(gradientNormalization)
    builder.gradientNormalizationThreshold(gradientNormalizationThreshold)
    l1?.let {
        builder.l1(it.weight)
        builder.l1Bias(it.bias)
    }
    l2?.let {
        builder.l2(it.weight)
        builder.l2Bias(it.bias)
    }
    builder.regularization.addAll(regularizations)
    builder.regularizationBias.addAll(regularizationBiases)
}

internal fun IBaseLayerConf.copyTo(baseLayerConf: IBaseLayerConf) {
    baseLayerConf.name = name
    baseLayerConf.activation = activation
    baseLayerConf.weightInitFunction = weightInitFunction
    baseLayerConf.biasInit = biasInit
    baseLayerConf.dropOut = dropOut
    baseLayerConf.weightNoise = weightNoise
    baseLayerConf.updater = updater
    baseLayerConf.biasUpdater = biasUpdater
    baseLayerConf.gradientNormalization = gradientNormalization
    baseLayerConf.gradientNormalizationThreshold = gradientNormalizationThreshold
    baseLayerConf.l1 = l1
    baseLayerConf.l2 = l2
    baseLayerConf.regularizations = regularizations
    baseLayerConf.regularizationBiases = regularizationBiases
}
