package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.layers.Layer

internal class BaseLayerConf : BaseLayer.Builder<BaseLayerConf>(), IBaseLayerConf {
	override var activation = super.activationFn

	override var weightInit = super.weightInit

	override var biasInit = super.biasInit

	override var dist = super.dist

	override var dropOut = super.iDropout

	override var weightNoise = super.weightNoise

	override var updater = super.iupdater

	override var biasUpdater = super.biasUpdater

	override var gradientNormalization = super.gradientNormalization

	override var gradientNormalizationThreshold = super.gradientNormalizationThreshold

	override var l1Weights = super.l1

	override var l1Bias = super.l1Bias

	override var l2Weights = super.l2

	override var l2Bias = super.l2Bias

	override fun <E : Layer?> build(): E {
		throw NotImplementedError()
	}
}
