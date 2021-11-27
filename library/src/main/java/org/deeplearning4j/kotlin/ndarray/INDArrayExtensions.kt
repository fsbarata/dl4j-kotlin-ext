package org.deeplearning4j.kotlin.ndarray

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.GreaterThan
import org.nd4j.linalg.indexing.conditions.LessThan

fun INDArray.coerceAtMost(maxValue: Double) = dup().also { BooleanIndexing.replaceWhere(it, maxValue, GreaterThan(maxValue)) }
fun INDArray.coerceAtLeast(minValue: Double) = dup().also { BooleanIndexing.replaceWhere(it, minValue, LessThan(minValue)) }
fun INDArray.coerceIn(minValue: Double, maxValue: Double) = dup().also {
	BooleanIndexing.replaceWhere(it, minValue, LessThan(minValue))
	BooleanIndexing.replaceWhere(it, maxValue, GreaterThan(maxValue))
}
