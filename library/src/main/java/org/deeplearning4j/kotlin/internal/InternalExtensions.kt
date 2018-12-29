package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.IFeedForwardLayerConf
import org.deeplearning4j.kotlin.layer.ILossLayerConf
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer


internal fun <T: BaseOutputLayer.Builder<T>> ILossLayerConf.applyTo(builder: T) {
	builder.lossFunction(lossFunction)
}

internal fun <T : FeedForwardLayer.Builder<T>> IFeedForwardLayerConf.applyTo(builder: T) {
	(this as IBaseLayerConf).applyTo(builder)
	builder.nIn(nIn)
	builder.nOut(nOut)
}

internal fun <T : BaseLayer.Builder<T>> IBaseLayerConf.applyTo(builder: T) {
	builder.activation(activation)
	builder.weightInit(weightInit)
	builder.biasInit(biasInit)
	builder.dist(dist)
	builder.dropOut(dropOut)
	builder.weightNoise(weightNoise)
	builder.updater(updater)
	builder.biasUpdater(biasUpdater)
	builder.gradientNormalization(gradientNormalization)
	builder.gradientNormalizationThreshold(gradientNormalizationThreshold)
	builder.l1(l1Weights)
	builder.l1Bias(l1Bias)
	builder.l2(l2Weights)
	builder.l2Bias(l2Bias)
}

internal fun IBaseLayerConf.copyTo(baseLayerConf: IBaseLayerConf) {
	baseLayerConf.activation = activation
	baseLayerConf.weightInit = weightInit
	baseLayerConf.biasInit = biasInit
	baseLayerConf.dist = dist
	baseLayerConf.dropOut = dropOut
	baseLayerConf.weightNoise = weightNoise
	baseLayerConf.updater = updater
	baseLayerConf.biasUpdater = biasUpdater
	baseLayerConf.gradientNormalization = gradientNormalization
	baseLayerConf.gradientNormalizationThreshold = gradientNormalizationThreshold
	baseLayerConf.l1Weights = l1Weights
	baseLayerConf.l1Bias = l1Bias
	baseLayerConf.l2Weights = l2Weights
	baseLayerConf.l2Bias = l2Bias
}


