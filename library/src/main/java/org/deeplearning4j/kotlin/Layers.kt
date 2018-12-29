package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.layer.DenseLayerConf
import org.deeplearning4j.kotlin.layer.OutputLayerConf

fun denseLayer(init: DenseLayerConf.() -> Unit) =
		DenseLayerConf().apply(init).build()

fun outputLayer(init: OutputLayerConf.() -> Unit) =
		OutputLayerConf().apply(init).build()

