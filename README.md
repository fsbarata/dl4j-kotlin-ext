# dl4j-kotlin-ext
dl4j kotlin extension is a DSL and extension methods library for writing Neural networks with [`DeepLearning4J`](https://github.com/deeplearning4j/deeplearning4j)

## Layers

Layers can be configurated by c DSL

    val layer = denseLayer {
        activation = Activation.SOFTMAX.activationFunction
        updater = RmsProp()
        nIn = 4
        nOut = 5
        hasBias = false
    }


