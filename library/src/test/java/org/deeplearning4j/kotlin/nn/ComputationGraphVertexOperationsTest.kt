package org.deeplearning4j.kotlin.nn

import org.deeplearning4j.kotlin.DIST
import org.deeplearning4j.kotlin.graph
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.junit.Assert.assertEquals
import org.junit.Test
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

class ComputationGraphVertexOperationsTest {
	@Test
	fun graph_plus() {
		buildGraphWithIdentityAndOnesLayers { identityLayer, onesLayers ->
			identityLayer + onesLayers
		}.assertOutput(
				expectedOutput = doubleArrayOf(5.0, 3.0, 7.0),
				input1 = doubleArrayOf(2.0, 0.0, 4.0),
				input2 = doubleArrayOf(1.0, 2.0)
		)
	}

	@Test
	fun graph_minus() {
		buildGraphWithIdentityAndOnesLayers { identityLayer, onesLayers ->
			identityLayer - onesLayers
		}.assertOutput(
				expectedOutput = doubleArrayOf(0.0, -2.0, -2.0),
				input1 = doubleArrayOf(2.0, 0.0, 0.0),
				input2 = doubleArrayOf(1.0, 1.0)
		)
	}

	@Test
	fun graph_times() {
		buildGraphWithIdentityAndOnesLayers { identityLayer, onesLayers ->
			identityLayer * onesLayers
		}.assertOutput(
				expectedOutput = doubleArrayOf(5.0, 0.0, 10.0),
				input1 = doubleArrayOf(2.0, 0.0, 4.0),
				input2 = doubleArrayOf(1.0, 1.5)
		)
	}

	@Test
	fun graph_max() {
		buildGraphWithIdentityAndOnesLayers { identityLayer, onesLayers ->
			max(identityLayer, onesLayers)
		}.assertOutput(
				expectedOutput = doubleArrayOf(3.0, 3.0, 4.0),
				input1 = doubleArrayOf(2.0, 0.0, 4.0),
				input2 = doubleArrayOf(1.0, 2.0)
		)
	}

	@Test
	fun graph_concat() {
		buildGraphWithIdentityAndOnesLayers { identityLayer, onesLayers ->
			concat(identityLayer, onesLayers)
		}.assertOutput(
				expectedOutput = doubleArrayOf(2.0, 0.0, 4.0, 3.0, 3.0, 3.0),
				input1 = doubleArrayOf(2.0, 0.0, 4.0),
				input2 = doubleArrayOf(1.0, 2.0)
		)
	}

	@Test
	fun graph_average() {
		buildGraphWithIdentityAndOnesLayers { identityLayer, onesLayers ->
			average(identityLayer, onesLayers)
		}.assertOutput(
				expectedOutput = doubleArrayOf(3.5, 2.5, 4.5),
				input1 = doubleArrayOf(2.0, 0.0, 4.0),
				input2 = doubleArrayOf(1.0, 4.0)
		)
	}

	companion object {
		val OPTIMIZATION_ALGORITHM = OptimizationAlgorithm.CONJUGATE_GRADIENT
		val LOSS_FUNCTION = LossFunctions.LossFunction.MSE.iLossFunction

		private fun buildComputationGraph(buildGraph: ComputationGraphConf.() -> VertexDescriptor) = graph {
			defaultConfig {
				optimizationAlgo = OPTIMIZATION_ALGORITHM
			}

			baseLayerConfig {
				activation = Activation.IDENTITY.activationFunction
				dist = DIST
			}

			lossLayer(buildGraph()) {
				lossFunction = LOSS_FUNCTION
			}
		}.also { it.init() }

		private fun buildGraphWithIdentityAndOnesLayers(combineFunction: ComputationGraphConf.(VertexDescriptor, VertexDescriptor) -> VertexDescriptor) =
				buildComputationGraph {
					val input = inputVertex()
					val input2 = inputVertex()

					val layerOne = denseLayer(input) {
						weightInit = WeightInit.IDENTITY
						nIn = 3
						nOut = 3
					}

					val layerTwo = denseLayer(input2) {
						weightInit = WeightInit.ONES
						nIn = 2
						nOut = 3
					}

					combineFunction(layerOne, layerTwo)
				}

		private fun ComputationGraph.assertOutput(expectedOutput: DoubleArray, input1: DoubleArray, input2: DoubleArray) {
			assertEquals(expectedOutput.toList(), outputSingle(Nd4j.create(input1), Nd4j.create(input2)).toDoubleVector().toList())
		}
	}
}