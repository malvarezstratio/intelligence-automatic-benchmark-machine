package com.stratio.intelligence.automaticBenchmark.models.logisticModelTree

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters
import org.apache.spark.mllib.tree.cost.{AccuracyCostFunction, LmtCostFunction}

case class LMTParams() extends ModelParameters{

  // · Impurity: Criterion used for information gain calculation.
  //      Classification -> Supported values: "gini" (recommended) or "entropy".
  // · MaxDepth: Maximum depth of the tree.
  //      E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (suggested value: 5)
  // · MaxBins: maximum number of bins used for splitting features (suggested value: 32)

  // Flag for enabling LMT debug information
  var debugConsole: Boolean = false
  var numClasses:Int = 2

  // Tree params
  var impurity:String = LMTParams.IMPURITY_GINI
  var maxDepth:Int    = 5
  var maxBins:Int     = 32

  // Maximum number of points for doing a local regression (using Weka)
  var maxPointsForLocalRegression: Int = 10000

  // Pruning type {validation,folds (CART)}
  var pruningType:String = LMTParams.PRUNING_TYPE_VALIDATION

  // Pruning
  var prune: String = "AUC" // {"AUC", "COST"}
    // · Cost function to use if prune type is "COST" -> f(confMatrix)
    var costFunction: LmtCostFunction = AccuracyCostFunction

  // Minimum gain required for allowing a new split ->
  //    The combined performance of the LRs in the children is better than the parent's LR performance
  var pruningRatio:Double = 0.1

  // TODO - If the number of instances in a node/one of the children ??? is lower, the prune is over
  var minElements:Int = -1

  // TODO Folds ??
  var seed:Long = -1
  var numFolds:Int = 5
  var weights : scala.Array[scala.Double] = Array(0.8,0.2)

  // Number of fold to train different LMTs when Pruning type = folds (CART)
  var numFoldsCart:Int = 5

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

