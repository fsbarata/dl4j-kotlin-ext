package org.deeplearning4j.kotlin.layer

import org.nd4j.linalg.lossfunctions.ILossFunction


interface ILossLayerConf {
	var lossFunction: ILossFunction
}
