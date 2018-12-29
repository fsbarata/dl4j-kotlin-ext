package org.deeplearning4j.kotlin.layer

import org.deeplearning4j.kotlin.internal.FeedForwardConf
import org.deeplearning4j.kotlin.internal.applyTo
import org.deeplearning4j.nn.conf.layers.DenseLayer


class DenseLayerConf : IFeedForwardLayerConf by FeedForwardConf() {
	internal val builder: DenseLayer.Builder = DenseLayer.Builder()

	var hasBias: Boolean
		@Deprecated("No getter", level = DeprecationLevel.ERROR) get() = throw NoSuchFieldError()
		set(value) {
			builder.hasBias(value)
		}

	fun build() = builder.also(this::applyTo).build()
}
