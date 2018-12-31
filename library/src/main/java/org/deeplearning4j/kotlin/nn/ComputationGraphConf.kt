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
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex
import org.deeplearning4j.nn.conf.graph.GraphVertex
import org.deeplearning4j.nn.conf.graph.LayerVertex
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.layers.util.IdentityLayer
import java.util.*

class ComputationGraphConf {
	private val builder = ComputationGraphConfiguration.GraphBuilder(NeuralNetConfiguration.Builder())

	private var defaultConfigBuilder = builder.globalConfiguration

	private var baseLayerConfig: IBaseLayerConf? = null

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

	fun inputVertex(name: String? = null): VertexDescriptor = VertexDescriptor(name.orRandomName()).also { builder.addInputs(it.name) }

	fun defaultConfig(init: NeuralNetConf.() -> Unit) {
		defaultConfigBuilder = NeuralNetConf().apply(init).builder
	}

	fun baseLayerConfig(init: IBaseLayerConf.() -> Unit) {
		baseLayerConfig = BaseLayerConf().apply(init)
	}

	fun vertex(inputs: List<VertexDescriptor>, name: String? = null, graphVertex: GraphVertex): VertexDescriptor =
			VertexDescriptor(name.orRandomName()).also { vertex ->
				vertices += vertex to VertexWithInputs(inputs, graphVertex)
			}

	fun vertex(vararg inputs: VertexDescriptor, name: String? = null, graphVertex: GraphVertex): VertexDescriptor =
			vertex(inputs.toList(), name, graphVertex)

	fun layer(vararg inputs: VertexDescriptor, inputPreProcessor: InputPreProcessor? = null, layer: Layer): VertexDescriptor =
			vertex(*inputs, name = layer.layerName, graphVertex = LayerVertex(defaultConfigBuilder.clone().layer(layer).build(), inputPreProcessor))
					.also { layer.layerName = it.name }

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


	operator fun VertexDescriptor.plus(other: VertexDescriptor) =
			vertex(this, other.duplicateIfEqualTo(this), graphVertex = ElementWiseVertex(ElementWiseVertex.Op.Add))

	operator fun VertexDescriptor.minus(other: VertexDescriptor) =
			vertex(this, other.duplicateIfEqualTo(this), graphVertex = ElementWiseVertex(ElementWiseVertex.Op.Subtract))

	operator fun VertexDescriptor.times(other: VertexDescriptor) =
			vertex(this, other.duplicateIfEqualTo(this), graphVertex = ElementWiseVertex(ElementWiseVertex.Op.Product))

	fun average(vararg vertices: VertexDescriptor, name: String? = null) =
			average(vertices.toList(), name)

	fun average(vertices: List<VertexDescriptor>, name: String? = null) =
			vertex(vertices, name, ElementWiseVertex(ElementWiseVertex.Op.Average))

	fun max(vararg vertices: VertexDescriptor, name: String? = null) =
			max(vertices.toList(), name)

	fun max(vertices: List<VertexDescriptor>, name: String? = null) =
			vertex(vertices, name, ElementWiseVertex(ElementWiseVertex.Op.Max))

	fun concat(vararg vertices: VertexDescriptor, name: String? = null) =
			concat(vertices.toList(), name)

	fun concat(vertices: List<VertexDescriptor>, name: String? = null) =
			vertex(vertices, name, MergeVertex())

	fun VertexDescriptor.duplicate() =
			layer(this, layer = IdentityLayer())

	fun build(): ComputationGraphConfiguration {
		vertices.forEach { (descriptor, vertexWithConnection) ->
			val (inputs, graphVertex) = vertexWithConnection
			builder.addVertex(descriptor.name, graphVertex, *inputs.names())
			Unit
		}

		builder.setOutputs(*outputs.names())

		return ComputationGraphConfiguration.GraphBuilder(builder.build(), defaultConfigBuilder).build()
	}

	private fun String?.orRandomName() = this ?: UUID.randomUUID().toString()

	private fun List<VertexDescriptor>.names() = map { it.name }.toTypedArray()

	private data class VertexWithInputs(
			val inputs: List<VertexDescriptor>,
			val vertex: GraphVertex
	)

	private fun VertexDescriptor.duplicateIfEqualTo(other: VertexDescriptor) =
			if (this == other) duplicate().also { println("Warning: Operation uses same vertex, duplicating") }
			else this
}