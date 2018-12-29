package org.deeplearning4j.kotlin.layer

import org.deeplearning4j.kotlin.internal.FeedForwardConf
import org.deeplearning4j.kotlin.internal.applyTo
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.lossfunctions.ILossFunction

class OutputLayerConf : IFeedForwardLayerConf by FeedForwardConf() {
	internal val builder: OutputLayer.Builder = OutputLayer.Builder()

	var hasBias: Boolean
		@Deprecated("No getter", level = DeprecationLevel.ERROR) get() = throw NoSuchFieldError()
		set(value) {
			builder.hasBias(value)
		}


	var lossFunction: ILossFunction
		@Deprecated("No getter", level = DeprecationLevel.ERROR) get() = throw NoSuchFieldError()
		set(value) {
			builder.lossFunction(value)
		}


	fun build() = builder.also(this::applyTo).build()
}
