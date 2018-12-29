package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.nn.VertexDescriptor
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.graph.LayerVertex
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.weights.WeightInit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

class NeuralNetworksTest {
	@Test
	fun multilayerNetwork_oneLayer_hasConfigurations() {
		val network = multilayerNetwork {
			defaultConfig {
				miniBatch = true
				optimizationAlgo = OPTIMIZATION_ALGORITHM
				trainingWorkspaceMode = WorkspaceMode.ENABLED
			}
			tbpttFwdLength = 1
			outputLayer {
				applyTestConstants()
				nIn = 1
				nOut = 2
				lossFunction = LOSS_FUNCTION
			}
		}

		assertTrue(network.defaultConfiguration.isMiniBatch)
		assertEquals(OPTIMIZATION_ALGORITHM, network.defaultConfiguration.optimizationAlgo)
		assertEquals(1, network.layerWiseConfigurations.tbpttFwdLength)

		val layer = network.layerWiseConfigurations.confs[0].layer as OutputLayer
		layer.assertTestConstants()
		assertEquals(LOSS_FUNCTION, layer.lossFn)
	}

	@Test
	fun multilayerNetwork_multipleLayer_withBaseConfiguration_isCorrect() {
		val network = multilayerNetwork {
			defaultConfig {
				optimizationAlgo = OPTIMIZATION_ALGORITHM
			}

			baseLayerConfig {
				applyTestConstants()
			}

			denseLayer {
				nIn = 3
				nOut = 5
			}
			outputLayer {
				nIn = 5
				nOut = 2
				lossFunction = LOSS_FUNCTION
			}
		}

		network.init()

		val layer1 = network.layerWiseConfigurations.confs[0].layer as DenseLayer
		layer1.assertTestConstants()
		val layer2 = network.layerWiseConfigurations.confs[1].layer as OutputLayer
		layer2.assertTestConstants()
	}

	@Test
	fun graph_configuresLayers() {
		lateinit var layerOne: VertexDescriptor
		lateinit var layerTwo: VertexDescriptor
		lateinit var outputLayer: VertexDescriptor

		val network = graph {
			defaultConfig {
				miniBatch = true
				optimizationAlgo = OPTIMIZATION_ALGORITHM
				trainingWorkspaceMode = WorkspaceMode.ENABLED
			}

			baseLayerConfig {
				applyTestConstants()
			}

			val input = inputVertex()

			layerOne = denseLayer(input) {
				name = "1"
				nIn = 3
				nOut = 5
			}

			layerTwo = denseLayer(layerOne, input) {
				nIn = 6
				nOut = 5
			}

			val layerThree = denseLayer(layerTwo) {
				nIn = 3
				nOut = 2
			}

			outputLayer = outputLayer(layerThree) {
				nIn = 2
				nOut = 5
				lossFunction = LOSS_FUNCTION
			}

			outputLayer(layerTwo) {
				nIn = 1
				nOut = 3
			}
		}

		network.init()

		assertTrue(network.configuration.defaultConfiguration.isMiniBatch)
		assertEquals(OPTIMIZATION_ALGORITHM, network.configuration.defaultConfiguration.optimizationAlgo)

		(network.configuration.vertices[layerOne.name] as LayerVertex).apply {
			(layerConf.layer as BaseLayer).assertTestConstants()
			assertEquals(3, (layerConf.layer as FeedForwardLayer).nIn)
			assertEquals("1", layerConf.layer.layerName)
		}
		(network.configuration.vertices[layerTwo.name] as LayerVertex).apply {
			(layerConf.layer as BaseLayer).assertTestConstants()
			assertEquals(6, (layerConf.layer as FeedForwardLayer).nIn)
		}
		(network.configuration.vertices[outputLayer.name] as LayerVertex).apply {
			(layerConf.layer as BaseLayer).assertTestConstants()
			assertEquals(2, (layerConf.layer as FeedForwardLayer).nIn)
			assertEquals(LOSS_FUNCTION, (layerConf.layer as BaseOutputLayer).lossFn)
		}

		assertEquals(2, network.configuration.networkOutputs.size)
	}

	@Test
	fun graph_plus() {
		lateinit var layerThree: VertexDescriptor

		val network = graph {
			defaultConfig {
				optimizationAlgo = OPTIMIZATION_ALGORITHM
			}

			baseLayerConfig {
				activation = Activation.IDENTITY.activationFunction
				dist = DIST
			}

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

			layerThree = layerOne + layerTwo

			lossLayer(layerThree) {
				lossFunction = LOSS_FUNCTION
			}
		}

		network.init()

		val input1 = Nd4j.ones(3)
		val input2 = Nd4j.ones(2)
		val output = network.outputSingle(input1, input2)

		val outputDoubles = output.toDoubleVector().toList()
		assertEquals(listOf(3.0, 3.0, 3.0), outputDoubles)
	}

	companion object {
		val OPTIMIZATION_ALGORITHM = OptimizationAlgorithm.CONJUGATE_GRADIENT
		val LOSS_FUNCTION = LossFunctions.LossFunction.MSE.iLossFunction
	}
}