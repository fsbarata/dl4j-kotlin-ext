package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.nn.ComputationGraphConf
import org.deeplearning4j.kotlin.nn.MultiLayerConf
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.buffer.DataType

fun multilayerNetwork(dataType: DataType, init: MultiLayerConf.() -> Unit) =
		MultiLayerNetwork(MultiLayerConf(dataType).apply(init).build())

fun graph(init: ComputationGraphConf.() -> Unit) =
		ComputationGraph(ComputationGraphConf().apply(init).build())
