package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.nn.MultiLayerConf
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

fun multilayerNetwork(init: MultiLayerConf.() -> Unit) =
		MultiLayerNetwork(MultiLayerConf().apply(init).build())

