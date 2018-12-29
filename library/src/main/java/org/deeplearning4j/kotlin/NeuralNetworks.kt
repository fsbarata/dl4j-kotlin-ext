package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.nn.ComputationGraphConf
import org.deeplearning4j.kotlin.nn.MultiLayerConf
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

fun multilayerNetwork(init: MultiLayerConf.() -> Unit) =
		MultiLayerNetwork(MultiLayerConf().apply(init).build())

fun graph(init: ComputationGraphConf.() -> Unit) =
		ComputationGraph(ComputationGraphConf().apply(init).build())
