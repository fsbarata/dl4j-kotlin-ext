package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.IFeedForwardLayerConf
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer
import org.deeplearning4j.nn.conf.layers.Layer


internal class FeedForwardConf : IFeedForwardLayerConf {
	override var nIn: Int = initialValues.nIn

	override var nOut: Int = initialValues.nOut
}

private val initialValues = FeedForwardLayerBuilderProxy()

private class FeedForwardLayerBuilderProxy : FeedForwardLayer.Builder<FeedForwardLayerBuilderProxy>() {
	val nIn = super.nIn
	val nOut = super.nOut

	override fun <E : Layer?> build(): E {
		throw NotImplementedError()
	}
}