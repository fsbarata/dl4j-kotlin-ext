package org.deeplearning4j.kotlin.ndarray

import org.deeplearning4j.kotlin.assertEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j

class INDArrayExtensionsKtTest {

	@Test
	fun coerceAtMost() {
		Nd4j.create(doubleArrayOf(1.1, -2.0, 0.0, 1.3, -4.0)).coerceAtMost(1.0).assertEquals(doubleArrayOf(1.0, -2.0, 0.0, 1.0, -4.0))
	}

	@Test
	fun coerceAtLeast() {
		Nd4j.create(doubleArrayOf(1.1, -2.0, 0.0, 1.3, -4.0)).coerceAtLeast(1.0).assertEquals(doubleArrayOf(1.1, 1.0, 1.0, 1.3, 1.0))
	}

	@Test
	fun coerceIn() {
		Nd4j.create(doubleArrayOf(1.1, -2.0, 0.0, 1.3, -4.0)).coerceIn(-1.0, 1.0).assertEquals(doubleArrayOf(1.0, -1.0, 0.0, 1.0, -1.0))
	}
}
