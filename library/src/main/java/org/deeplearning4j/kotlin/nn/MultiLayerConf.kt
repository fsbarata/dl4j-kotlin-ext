package org.deeplearning4j.kotlin.nn

import org.deeplearning4j.kotlin.denseLayer
import org.deeplearning4j.kotlin.internal.BaseLayerConf
import org.deeplearning4j.kotlin.layer.DenseLayerConf
import org.deeplearning4j.kotlin.layer.IBaseLayerConf
import org.deeplearning4j.kotlin.layer.LossLayerConf
import org.deeplearning4j.kotlin.layer.OutputLayerConf
import org.deeplearning4j.kotlin.lossLayer
import org.deeplearning4j.kotlin.outputLayer
import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.Layer
import org.nd4j.linalg.api.buffer.DataType

class MultiLayerConf(var dataType: DataType) {
    private val builder: MultiLayerConfiguration.Builder = MultiLayerConfiguration.Builder()

    private var defaultConfigBuilder = NeuralNetConfiguration.Builder()

    private var baseLayerConfig: IBaseLayerConf? = null

    private var inputPreProcessors = emptyMap<Int, InputPreProcessor>()
    private var layers = emptyList<Layer>()

    fun defaultConfig(init: NeuralNetConf.() -> Unit) {
        defaultConfigBuilder = NeuralNetConf().apply(init).builder
    }

    var backpropType
        get() = builder.backpropType
        set(value) {
            builder.backpropType(value)
        }
    var tbpttFwdLength
        get() = builder.tbpttFwdLength
        set(value) {
            builder.tBPTTForwardLength(value)
        }
    var tbpttBackLength
        get() = builder.tbpttBackLength
        set(value) {
            builder.tBPTTBackwardLength(value)
        }

    var inputType: InputType?
        get() = builder.inputType
        set(value) {
            builder.inputType = value
        }

    var validateOutputConfig
        get() = builder.isValidateOutputConfig
        set(value) {
            builder.validateOutputLayerConfig(value)
        }
    var trainingWorkspaceMode
        get() = builder.trainingWorkspaceMode
        set(value) {
            builder.trainingWorkspaceMode = value
        }
    var inferenceWorkspaceMode
        get() = builder.inferenceWorkspaceMode
        set(value) {
            builder.inferenceWorkspaceMode = value
        }
    var cacheMode
        get() = builder.cacheMode
        set(value) {
            builder.cacheMode(value)
        }

    fun baseLayerConfig(init: IBaseLayerConf.() -> Unit) {
        baseLayerConfig = BaseLayerConf().apply(init)
    }

    fun denseLayer(init: DenseLayerConf.() -> Unit) =
        layer(denseLayer(baseLayerConfig, init))

    fun outputLayer(init: OutputLayerConf.() -> Unit) =
        layer(outputLayer(baseLayerConfig, init))

    fun lossLayer(init: LossLayerConf.() -> Unit) =
        layer(lossLayer(baseLayerConfig, init))

    fun layer(layer: Layer, inputPreProcessor: InputPreProcessor? = null) {
        if (inputPreProcessor != null) inputPreProcessors += layers.size to inputPreProcessor
        layers += layer
    }

    fun build(): MultiLayerConfiguration {
        builder.inputPreProcessors = inputPreProcessors
        builder.confs = layers.map { defaultConfigBuilder.clone().layer(it).build() }
		builder.dataType(dataType)
        return builder.build()
    }
}
