package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.IFeedForwardLayerConf
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer
import org.deeplearning4j.nn.conf.layers.Layer


internal class FeedForwardConf
	: FeedForwardLayer.Builder<FeedForwardConf>(), IBaseLayerConf by BaseLayerConf(), IFeedForwardLayerConf {
	override var nIn: Int = super.nIn
	override var nOut: Int = super.nOut

	override fun <E : Layer?> build(): E {
		throw NotImplementedError()
	}
}