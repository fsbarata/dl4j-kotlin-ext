package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.NoOp

internal class BaseLayerConf : IBaseLayerConf {
	override var name: String? = null

	override var activation = initialValues.activation

	override var weightInit = initialValues.weightInit

	override var biasInit = initialValues.biasInit

	override var dist = initialValues.dist

	override var dropOut = initialValues.dropOut

	override var weightNoise = initialValues.weightNoise

	override var updater = initialValues.updater ?: NoOp()

	override var biasUpdater = initialValues.biasUpdater

	override var gradientNormalization = initialValues.gradientNormalization ?: GradientNormalization.None

	override var gradientNormalizationThreshold = initialValues.gradientNormalizationThreshold

	override var l1Weights = initialValues.l1Weights

	override var l1Bias = initialValues.l1Bias

	override var l2Weights = initialValues.l2Weights

	override var l2Bias = initialValues.l2Bias
}

private val initialValues = BaseLayerBuilderProxy()

private class BaseLayerBuilderProxy : BaseLayer.Builder<BaseLayerBuilderProxy>() {
	val activation = super.activationFn ?: Activation.RELU.activationFunction

	val weightInit = super.weightInit ?: WeightInit.XAVIER

	val biasInit = super.biasInit

	val dist = super.dist

	val dropOut = super.iDropout

	val weightNoise = super.weightNoise

	val updater = super.iupdater

	val biasUpdater = super.biasUpdater

	val gradientNormalization = super.gradientNormalization

	val gradientNormalizationThreshold = super.gradientNormalizationThreshold

	val l1Weights = super.l1

	val l1Bias = super.l1Bias

	val l2Weights = super.l2

	val l2Bias = super.l2Bias

	override fun <E : Layer?> build(): E {
		throw NotImplementedError()
	}
}
