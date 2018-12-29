package org.deeplearning4j.kotlin.nn

import org.deeplearning4j.nn.conf.BackpropType

sealed class BackpropStrategy(val type: BackpropType) {
	object Standard: BackpropStrategy(BackpropType.Standard)

	data class Truncated(
			val forwardLength: Int,
			val backwardLength: Int = forwardLength
	): BackpropStrategy(BackpropType.TruncatedBPTT)
}
