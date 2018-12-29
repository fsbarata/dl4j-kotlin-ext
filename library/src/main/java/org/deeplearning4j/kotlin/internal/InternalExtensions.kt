package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.IFeedForwardLayerConf
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer


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


