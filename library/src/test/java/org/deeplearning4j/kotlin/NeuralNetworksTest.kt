package org.deeplearning4j.kotlin

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NeuralNetworksTest {
	@Test
	fun multilayerNetwork_oneLayer_hasConfigurations() {
		val network = multilayerNetwork {
			defaultConfig {
				miniBatch = true
				optimizationAlgo = OptimizationAlgorithm.LBFGS
				trainingWorkspaceMode = WorkspaceMode.ENABLED
			}
			tbpttFwdLength = 1
			outputLayer {
				applyTestConstants()
				nIn = 1
				nOut = 2
			}
		}

		assertTrue(network.defaultConfiguration.isMiniBatch)
		assertEquals(OptimizationAlgorithm.LBFGS, network.defaultConfiguration.optimizationAlgo)
		assertEquals(1, network.layerWiseConfigurations.tbpttFwdLength)

		val layer = network.layerWiseConfigurations.confs[0].layer as OutputLayer
		layer.assertTestConstants()
	}

	@Test
	fun multilayerNetwork_multipleLayer_withBaseConfiguration_isCorrect() {
		val network = multilayerNetwork {
			defaultConfig {
				optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
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
			}
		}

		assertEquals(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, network.defaultConfiguration.optimizationAlgo)

		val layer1 = network.layerWiseConfigurations.confs[0].layer as DenseLayer
		layer1.assertTestConstants()
		val layer2 = network.layerWiseConfigurations.confs[1].layer as OutputLayer
		layer2.assertTestConstants()
	}
}