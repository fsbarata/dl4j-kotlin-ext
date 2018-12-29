# dl4j-kotlin-ext
dl4j-kotlin-ext is an unofficial kotlin DSL and extension methods library for writing Neural networks with [`DeepLearning4j`](https://github.com/deeplearning4j/deeplearning4j)

## Motivation
If you are a kotlin user, you may be struggling with the boilerplate needed to build networks and layers and may not always understand what fits together. In this library you will find a richer way of dealing with layers, multilayer networks and computation graphs.

## Layers

Layers can be configurated by DSL. Here's an example of a DenseLayer:

    val layer = denseLayer {
        activation = Activation.SOFTMAX.activationFunction
        updater = RmsProp()
        nIn = 4
        nOut = 5
        hasBias = false
    }


