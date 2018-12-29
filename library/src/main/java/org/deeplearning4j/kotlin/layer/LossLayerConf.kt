package org.deeplearning4j.kotlin.layer

import org.deeplearning4j.kotlin.internal.BaseLayerConf
import org.deeplearning4j.kotlin.internal.BaseLossLayerConf
import org.deeplearning4j.kotlin.internal.applyTo
import org.deeplearning4j.nn.conf.layers.LossLayer

class LossLayerConf : ILossLayerConf by BaseLossLayerConf(), IBaseLayerConf by BaseLayerConf() {
	private val builder: LossLayer.Builder = LossLayer.Builder()

	fun build() = builder.also {
		(this as IBaseLayerConf).applyTo(builder)
		(this as ILossLayerConf).applyTo(builder)
	}.build()
}
