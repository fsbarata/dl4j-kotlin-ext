package org.deeplearning4j.kotlin

import org.junit.Assert
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


fun INDArray.assertEquals(expected: DoubleArray, epsilon: Double = Nd4j.EPS_THRESHOLD) {
	val result = toDoubleVector()
	val expectedList = expected.toList()
	val resultList = result.toList()
	Assert.assertEquals(
			"Size of arrays does not match ($expectedList vs $resultList)",
			expected.size, result.size)
	for (index in 0 until expected.size) {
		Assert.assertEquals(
				"Element at index $index does not match ($expectedList vs $resultList)",
				expected[index],
				result[index],
				epsilon
		)
	}
}
