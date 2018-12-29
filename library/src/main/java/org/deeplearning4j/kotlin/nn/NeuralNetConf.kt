package org.deeplearning4j.kotlin.nn

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.CacheMode
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.layers.Layer


class NeuralNetConf {
	internal val builder = NeuralNetConfiguration.Builder()

	var miniBatch: Boolean
		get() = builder.isMiniBatch
		set(value) {
			builder.miniBatch(value)
		}
	var maxNumLineSearchIterations: Int
		get() = builder.maxNumLineSearchIterations
		set(value) {
			builder.maxNumLineSearchIterations(value)
		}
	var seed: Long
		get() = builder.seed
		set(value) {
			builder.seed(value)
		}
	var optimizationAlgo: OptimizationAlgorithm
		get() = builder.optimizationAlgo
		set(value) {
			builder.optimizationAlgo(value)
		}
	var minimize: Boolean
		get() = builder.isMinimize
		set(value) {
			builder.minimize(value)
		}
	var trainingWorkspaceMode: WorkspaceMode
		get() = builder.trainingWorkspaceMode
		set(value) {
			builder.trainingWorkspaceMode(value)
		}
	var inferenceWorkspaceMode: WorkspaceMode
		get() = builder.inferenceWorkspaceMode
		set(value) {
			builder.inferenceWorkspaceMode(value)
		}
	var cacheMode: CacheMode
		get() = builder.cacheMode
		set(value) {
			builder.cacheMode(value)
		}

	fun build() = builder.build()
}
