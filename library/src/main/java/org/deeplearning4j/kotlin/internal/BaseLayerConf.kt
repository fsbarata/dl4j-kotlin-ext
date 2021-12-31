package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.WeightBias
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.weights.IWeightInit
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.learning.config.NoOp

internal class BaseLayerConf : IBaseLayerConf {
    override var name: String? = null

    override var activation = initialValues.activation

    override var weightInitFunction = initialValues.weightInitialization

    override var biasInit = initialValues.biasInitValues

    override var dropOut = initialValues.dropOut

    override var weightNoise = initialValues.weightNoise

    override var updater = initialValues.updater ?: NoOp()

    override var biasUpdater = initialValues.gradientUpdater

    override var gradientNormalization = initialValues.gradientNorm ?: GradientNormalization.None

    override var gradientNormalizationThreshold = initialValues.gradientNormThreshold

    override var regularizations = initialValues.regularization
    override var regularizationBiases = initialValues.regularizationBias

    override var l1: WeightBias? = null
    override var l2: WeightBias? = null
}

private val initialValues = BaseLayerBuilderProxy()

private class BaseLayerBuilderProxy : BaseLayer.Builder<BaseLayerBuilderProxy>() {
    val activation = super.activationFn

    val weightInitialization: IWeightInit = super.weightInitFn ?: WeightInit.XAVIER.weightInitFunction

    val biasInitValues = super.biasInit

    val dropOut = super.iDropout

    val wgtNoise = super.weightNoise

    val updater = super.iupdater

    val gradientUpdater = super.biasUpdater

    val gradientNorm = super.gradientNormalization

    val gradientNormThreshold = super.gradientNormalizationThreshold

    val regularizations = super.regularization
    val regularizationBiases = super.regularizationBias

    override fun <E : Layer?> build(): E {
        throw NotImplementedError()
    }
}
