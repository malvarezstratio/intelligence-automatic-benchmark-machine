package com.stratio.intelligence.automaticBenchmark.models.logisticModelTree

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters
import org.apache.spark.mllib.tree.cost.{AccuracyCostFunction, LmtCostFunction}

case class LMTParams() extends ModelParameters{

  // · Impurity: Criterion used for information gain calculation.
  //      Classification -> Supported values: "gini" (recommended) or "entropy".
  // · MaxDepth: Maximum depth of the tree.
  //      E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (suggested value: 5)
  // · MaxBins: maximum number of bins used for splitting features (suggested value: 32)

  var debugConsole: Boolean = false
  var numClasses:Int = 2

  // Tree params
  var impurity:String = LMTParams.IMPURITY_GINI
  var maxDepth:Int = 5
  var maxBins:Int = 32

  // Pruning
  var prune: String = "AUC"
  var pruningType:String = LMTParams.PRUNING_TYPE_VALIDATION
  var pruningRatio:Double = 0.1

  var minElements:Int = 2000
  var numLocalRegression: Int = 10

  var costFunction: LmtCostFunction = AccuracyCostFunction

  // Folds
  var seed:Long = -1
  var weights : scala.Array[scala.Double] = Array(0.8,0.2)
  var numFolds:Int = 5
  var numFoldsRegression:Int = 5

}

object LMTParams{

  // Impurity
    val IMPURITY_ENTROPY        = "entropy"
    val IMPURITY_VARIANCE       = "variance"
    val IMPURITY_GINI           = "gini"

  // Pruning type
    val PRUNING_TYPE_VALIDATION = "validation"
    val PRUNING_TYPE_FOLDS      = "folds"

}

