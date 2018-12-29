package org.deeplearning4j.kotlin

import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.IFeedForwardLayerConf
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer
import org.junit.Assert.*
import org.junit.Test
import org.nd4j.linalg.lossfunctions.LossFunctions

class LayersTest {
	@Test
	fun denseLayer_config_layerHasParameters() {
		val layer1 = denseLayer {
			applyTestConstants()
			hasBias = true
		}
		val layer2 = denseLayer {
			applyTestConstants()
			hasBias = false
		}

		layer1.assertTestConstants()
		assertTrue(layer1.isHasBias)
		layer2.assertTestConstants()
		assertFalse(layer2.isHasBias)
	}

	@Test
	fun outputLayer_config_layerHasParameters() {
		val lossFunction1 = LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR.iLossFunction
		val lossFunction2 = LossFunctions.LossFunction.MSE.iLossFunction

		val layer1 = outputLayer {
			applyTestConstants()
			hasBias = true
			lossFunction = lossFunction1
		}
		val layer2 = outputLayer {
			applyTestConstants()
			hasBias = false
			lossFunction = lossFunction2
		}

		layer1.assertTestConstants()
		assertTrue(layer1.isHasBias)
		assertEquals(lossFunction1, layer1.lossFn)
		layer2.assertTestConstants()
		assertFalse(layer2.isHasBias)
		assertEquals(lossFunction2, layer2.lossFn)
	}

	private fun IFeedForwardLayerConf.applyTestConstants() {
		(this as IBaseLayerConf).applyTestConstants()
		nIn = INPUT_NODES
		nOut = OUTPUT_NODES
	}

	private fun <T : FeedForwardLayer> T.assertTestConstants() {
		(this as BaseLayer).assertTestConstants()
		assertEquals(INPUT_NODES.toLong(), nIn)
		assertEquals(OUTPUT_NODES.toLong(), nOut)
	}
}