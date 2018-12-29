package org.deeplearning4j.kotlin.internal

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.ILossLayerConf
import org.deeplearning4j.nn.conf.layers.LossLayer
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions


internal class BaseLossLayerConf : IBaseLayerConf by BaseLayerConf(), ILossLayerConf {
	override var lossFunction: ILossFunction = initialValues.lossFunction
}

private val initialValues = LossLayerBuilderProxy()

private class LossLayerBuilderProxy : LossLayer.Builder() {
	val lossFunction = super.lossFn
}

