package com.stratio.intelligence.automaticBenchmark.models.logisticModelTree

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters
import org.apache.spark.mllib.tree.cost.{AccuracyCostFunction, LmtCostFunction}

case class LMTParams() extends ModelParameters{

  // · Impurity: Criterion used for information gain calculation.
  //      Classification -> Supported values: "gini" (recommended) or "entropy".
  // · MaxDepth: Maximum depth of the tree.
  //      E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (suggested value: 5)
  // · MaxBins: maximum number of bins used for splitting features (suggested value: 32)

  val numClasses:Int = 2

  // Debug
    private[this] var _debugConsole: Boolean = false
    def debugConsole: Boolean = _debugConsole
    def setDebugConsole(value: Boolean): LMTParams = { _debugConsole = value; this }

  // Tree params
    private[this] var _impurity: String = LMTParams.IMPURITY_GINI
    def impurity: String = _impurity
    def setImpurity(value: String): LMTParams = { _impurity = value; this }

    private[this] var _maxDepth: Int = 5
    def maxDepth: Int = _maxDepth;
    def setMaxDepth(value: Int): LMTParams = { _maxDepth = value; this }

    private[this] var _maxBins: Int = 32
    def maxBins: Int = _maxBins
    def setMaxBins(value: Int): LMTParams = { _maxBins = value;this }

  // Maximum number of points for doing a local regression (using Weka)
    private[this] var _maxPointsForLocalRegression: Int = 10000
    def maxPointsForLocalRegression: Int = _maxPointsForLocalRegression
    def setMaxPointsForLocalRegression(value: Int): LMTParams = { _maxPointsForLocalRegression = value; this }

  // Pruning type {validation,folds (CART)}
    private[this] var _pruningType: String = LMTParams.PRUNING_TYPE_VALIDATION
    def pruningType: String = _pruningType
    def setPruningType(value: String): LMTParams = { _pruningType = value; this }

  // Pruning
    private[this] var _prune: String = "AUC" // {"AUC", "COST"}
    def prune: String = _prune
    def setPrune(value: String): LMTParams = { _prune = value; this }
    // · Cost function to use if prune type is "COST" -> f(confMatrix)
      private[this] var _costFunction: LmtCostFunction = AccuracyCostFunction
      def costFunction: LmtCostFunction = _costFunction
      def setCostFunction(value: LmtCostFunction): LMTParams = { _costFunction = value; this }

  // Minimum gain required for allowing a new split ->
  //    The combined performance of the LRs in the children is better than the parent's LR performance
    private[this] var _pruningRatio: Double = 0.1
    def pruningRatio: Double = _pruningRatio
    def setPruningRatio(value: Double): LMTParams = { _pruningRatio = value;this }

  // TODO - If the number of instances in a node/one of the children ??? is lower, the prune is over
    private[this] var _minElements: Int = -1
    def minElements: Int = _minElements
    def setMinElements(value: Int): LMTParams = { _minElements = value; this }

  // TODO Folds ??
    private[this] var _seed: Long = -1
    def seed: Long = _seed
    def setSeed(value: Long): LMTParams = { _seed = value; this }

    private[this] var _numFolds: Int = 5
    def numFolds: Int = _numFolds
    def setNumFolds(value: Int): LMTParams = { _numFolds = value; this }

    private[this] var _weights: Array[Double] = Array(0.8,0.2)
    def weights: Array[Double] = _weights
    def setWeights(value: Array[Double]): LMTParams = { _weights = value; this }

    private[this] var _numFoldsCart: Int = 5
    def numFoldsCart: Int = _numFoldsCart
    def setNumFoldsCart(value: Int): LMTParams = { _numFoldsCart = value; this }

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

