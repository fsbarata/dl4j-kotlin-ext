package org.deeplearning4j.kotlin.nn

import org.deeplearning4j.kotlin.denseLayer
import org.deeplearning4j.kotlin.internal.BaseLayerConf
import org.deeplearning4j.kotlin.layer.DenseLayerConf
import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.LossLayerConf
import org.deeplearning4j.kotlin.layer.OutputLayerConf
import org.deeplearning4j.kotlin.lossLayer
import org.deeplearning4j.kotlin.outputLayer
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.GraphVertex
import org.deeplearning4j.nn.conf.graph.LayerVertex
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer
import org.deeplearning4j.nn.conf.layers.Layer

class ComputationGraphConf {
	private var builder = ComputationGraphConfiguration.GraphBuilder(NeuralNetConfiguration.Builder())

	private var defaultConfigBuilder = builder.globalConfiguration

	private var baseLayerConfig: IBaseLayerConf? = null

	private var count: Int = 0

	private var vertices = emptyMap<VertexDescriptor, VertexWithInputs>()
	private var outputs = emptyList<VertexDescriptor>()

	var allowDisconnected: Boolean
		get() = builder.isAllowDisconnected
		set(value) {
			builder.allowDisconnected(value)
		}
	var allowNoOutput: Boolean
		get() = builder.isAllowNoOutput
		set(value) {
			builder.allowNoOutput(value)
		}

	var backpropStrategy: BackpropStrategy
		get() = when (builder.backpropType) {
			BackpropType.Standard -> BackpropStrategy.Standard
			BackpropType.TruncatedBPTT -> BackpropStrategy.Truncated(builder.tbpttFwdLength, builder.tbpttBackLength)
			null -> throw NullPointerException()
		}
		set(value) {
			builder.backpropType(value.type)
			builder.tBPTTForwardLength(when (value) {
				is BackpropStrategy.Truncated -> value.forwardLength
				else -> builder.tbpttFwdLength
			})
			builder.tBPTTBackwardLength(when (value) {
				is BackpropStrategy.Truncated -> value.backwardLength
				else -> builder.tbpttBackLength
			})
		}

	var validateOutputConfig: Boolean
		get() = builder.isValidateOutputConfig
		set(value) {
			builder.validateOutputLayerConfig(value)
		}

	fun input(): VertexDescriptor = nextVertex().also { builder.addInputs(it) }

	fun defaultConfig(init: NeuralNetConf.() -> Unit) {
		defaultConfigBuilder = NeuralNetConf().apply(init).builder
	}

	fun baseLayerConfig(init: IBaseLayerConf.() -> Unit) {
		baseLayerConfig = BaseLayerConf().apply(init)
	}

	fun vertex(vararg inputs: VertexDescriptor, graphVertex: GraphVertex): VertexDescriptor =
			nextVertex().also { vertex ->
				vertices += vertex to VertexWithInputs(inputs.toList(), graphVertex)
			}

	fun layer(vararg inputs: VertexDescriptor, inputPreProcessor: InputPreProcessor? = null, layer: Layer): VertexDescriptor =
			vertex(*inputs, graphVertex = LayerVertex(defaultConfigBuilder.clone().layer(layer).build(), inputPreProcessor))

	fun denseLayer(vararg inputs: VertexDescriptor, inputPreProcessor: InputPreProcessor? = null, denseLayerConfig: DenseLayerConf.() -> Unit) =
			layer(*inputs, inputPreProcessor = inputPreProcessor, layer = denseLayer(baseLayerConfig, denseLayerConfig))

	fun output(vararg inputs: VertexDescriptor, inputPreProcessor: InputPreProcessor? = null, outputLayer: BaseOutputLayer) =
			layer(*inputs, inputPreProcessor = inputPreProcessor, layer = outputLayer)
					.also { outputs += it }

	fun lossLayer(vararg inputs: VertexDescriptor, inputPreProcessor: InputPreProcessor? = null, lossLayerConfig: LossLayerConf.() -> Unit) =
			layer(*inputs, inputPreProcessor = inputPreProcessor, layer = lossLayer(baseLayerConfig, lossLayerConfig))
					.also { outputs += it }

	fun outputLayer(vararg inputs: VertexDescriptor, inputPreProcessor: InputPreProcessor? = null, ouputLayerConfig: OutputLayerConf.() -> Unit) =
			output(*inputs, inputPreProcessor = inputPreProcessor, outputLayer = outputLayer(baseLayerConfig, ouputLayerConfig))


	fun build(): ComputationGraphConfiguration {
		vertices.forEach { (descriptor, vertexWithConnection) ->
			val (inputs, graphVertex) = vertexWithConnection
			builder.addVertex(descriptor, graphVertex, *(inputs.toTypedArray()))
		}

		builder.setOutputs(*outputs.toTypedArray())

		builder = ComputationGraphConfiguration.GraphBuilder(builder.build(), defaultConfigBuilder)

		return builder.build()
	}

	private fun nextVertex() = "$count".also { count++ }

	private data class VertexWithInputs(
			val inputs: List<String>,
			val vertex: GraphVertex
	)
}

typealias VertexDescriptor = String
