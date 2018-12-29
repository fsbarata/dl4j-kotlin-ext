package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.internal.copyTo
import org.deeplearning4j.kotlin.layer.DenseLayerConf
import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.LossLayerConf
import org.deeplearning4j.kotlin.layer.OutputLayerConf

fun denseLayer(baseLayerConf: IBaseLayerConf? = null, init: DenseLayerConf.() -> Unit) =
		DenseLayerConf().copyBaseAndApply(baseLayerConf, init).build()

fun outputLayer(baseLayerConf: IBaseLayerConf? = null, init: OutputLayerConf.() -> Unit) =
		OutputLayerConf().copyBaseAndApply(baseLayerConf, init).build()

fun lossLayer(baseLayerConf: IBaseLayerConf? = null, init: LossLayerConf.() -> Unit) =
		LossLayerConf().copyBaseAndApply(baseLayerConf, init).build()


private fun <T : IBaseLayerConf> T.copyBaseAndApply(baseLayerConf: IBaseLayerConf? = null, init: T.() -> Unit) =
		also { baseLayerConf?.copyTo(it) }.apply(init)