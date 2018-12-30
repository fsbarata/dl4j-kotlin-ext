package org.deeplearning4j.kotlin.ndarray

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.INDArrayIndex

operator fun INDArray.plus(operand: INDArray) = add(operand)
operator fun INDArray.minus(operand: INDArray) = sub(operand)
operator fun INDArray.times(operand: INDArray) = mmul(operand)
operator fun INDArray.rem(operand: INDArray) = fmod(operand)

operator fun INDArray.plus(scalar: Number) = add(scalar)
operator fun INDArray.minus(scalar: Number) = sub(scalar)
operator fun INDArray.times(scalar: Number) = mul(scalar)
operator fun INDArray.rem(scalar: Number) = fmod(scalar)

operator fun Number.plus(array: INDArray) = array + this
operator fun Number.minus(array: INDArray) = array.rsub(this)
operator fun Number.times(array: INDArray) = array * this
operator fun Number.div(array: INDArray) = array.rdiv(this)

operator fun INDArray.unaryMinus() = neg()

operator fun INDArray.plusAssign(operand: INDArray) {
	addi(operand)
}

operator fun INDArray.minusAssign(operand: INDArray) {
	subi(operand)
}

operator fun INDArray.timesAssign(operand: INDArray) {
	muli(operand)
}

operator fun INDArray.divAssign(operand: INDArray) {
	divi(operand)
}

operator fun INDArray.remAssign(operand: INDArray) {
	fmodi(operand)
}

operator fun INDArray.plusAssign(scalar: Number) {
	addi(scalar)
}

operator fun INDArray.minusAssign(scalar: Number) {
	subi(scalar)
}

operator fun INDArray.timesAssign(scalar: Number) {
	muli(scalar)
}

operator fun INDArray.divAssign(scalar: Number) {
	divi(scalar)
}

operator fun INDArray.remAssign(scalar: Number) {
	fmodi(scalar)
}

operator fun INDArray.get(vararg indices: Int) =
		getScalar(*indices)

operator fun INDArray.get(vararg indices: Long) =
		getScalar(*indices)

operator fun INDArray.set(vararg indices: INDArrayIndex, element: INDArray) {
	put(indices, element)
}

operator fun INDArray.set(vararg indices: INDArrayIndex, element: Number) {
	put(indices, element)
}

operator fun INDArray.set(indices: INDArray, element: INDArray) {
	put(indices, element)
}

@Deprecated("Unable to use long indices, converting to Int")
operator fun INDArray.set(vararg indices: Long, scalar: INDArray) {
	put(indices.toIntArray(), scalar)
}

operator fun INDArray.set(vararg indices: Int, scalar: INDArray) {
	put(indices, scalar)
}

operator fun INDArray.set(vararg indices: Long, scalar: Double) {
	putScalar(indices, scalar)
}

operator fun INDArray.set(vararg indices: Int, scalar: Double) {
	putScalar(indices, scalar)
}

operator fun INDArray.set(vararg indices: Long, scalar: Float) {
	putScalar(indices, scalar)
}

operator fun INDArray.set(vararg indices: Int, scalar: Float) {
	putScalar(indices, scalar)
}


@Deprecated("Loss of resolution")
private fun LongArray.toIntArray() = IntArray(size) { this[it].toInt() }
