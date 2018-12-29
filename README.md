[![](https://jitpack.io/v/fsbarata/dl4j-kotlin-ext.svg)](https://jitpack.io/#fsbarata/dl4j-kotlin-ext)

# dl4j-kotlin-ext
dl4j-kotlin-ext is an unofficial kotlin DSL and extension methods library for writing Neural networks with [`DeepLearning4j`](https://github.com/deeplearning4j/deeplearning4j)

## Motivation
If you are a kotlin user, you may be struggling with the boilerplate needed to build networks and layers and may not always understand what fits together. In this library you will find a richer way of dealing with layers, multilayer networks and computation graphs.


## Multilayer networks

Networks can be configured by DSL.

    val network = multilayerNetwork {
		defaultConfig {
			optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT
		}

		baseLayerConfig {
        	activation = Activation.SOFTMAX.activationFunction
	        weightInit = WeightInit.XAVIER
            updater = RmsProp()
		}

		denseLayer {
			nIn = 3
			nOut = 5
		}
		outputLayer {
			nIn = 5
			nOut = 2
			lossFunction = LossFunctions.LossFunction.MSE.iLossFunction
		}
	}
    ...

## Computation graphs

A computation graph can be configured like this:

    graph {
		defaultConfig {
			miniBatch = true
			optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT
			trainingWorkspaceMode = WorkspaceMode.ENABLED
		}

		baseLayerConfig {
        	activation = Activation.SOFTMAX.activationFunction
	        weightInit = WeightInit.XAVIER
            updater = RmsProp()
		}

		val input = inputVertex()

		val layerOne = denseLayer(input) {
			name = "1"
			nIn = 3
			nOut = 5
		}

		val layerTwo = denseLayer(layerOne, input) {
			nIn = 6
			nOut = 5
		}

		val layerThree = denseLayer(layerTwo) {
			nIn = 3
			nOut = 2
		}

		outputLayer(layerThree) {
			nIn = 2
			nOut = 5
			lossFunction = LossFunctions.LossFunction.MSE.iLossFunction
		}

		outputLayer(layerTwo) {
			nIn = 1
			nOut = 3
		}
	}
    
### Operators

When configuring a graph, a vertex can be added with an operator to combine multiple layers. This will create a subtract ElementWise operation:

    graph {
        ...
        val layerOne = denseLayer...
        val layerTwo = denseLayer...
        
        val layerThree = layerOne - layerTwo
        ...
    }

#### Available Operations
- plus (+) <=> Op.Add
- minus (-) <=> Op.Subtract
- times (*) <=> Op.Product
- average(input1,...) <=> Op.Average
- max(input1,...) <=> Op.Max
- concat(input1,...) <=> MergeVertex


## Layers

Layers can also be configured standalone by DSL.

    val layer = denseLayer {
        name = "layer1"
        activation = Activation.SOFTMAX.activationFunction
        updater = RmsProp()
        nIn = 4
        nOut = 5
        hasBias = false
    }

#### Available layers
- denseLayer
- outputLayer
- lossLayer
- WIP
