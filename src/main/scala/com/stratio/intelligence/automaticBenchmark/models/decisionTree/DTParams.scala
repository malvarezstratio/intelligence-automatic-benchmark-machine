package com.stratio.intelligence.automaticBenchmark.models.decisionTree

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters

case class DTParams() extends ModelParameters{

  // · Impurity: Criterion used for information gain calculation.
  //      Classification -> Supported values: "gini" (recommended) or "entropy".
  // · MaxDepth: Maximum depth of the tree.
  //      E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (suggested value: 5)
  // · MaxBins: maximum number of bins used for splitting features (suggested value: 32)

  private[this] var _impurity: String = DTParams.IMPURITY_GINI
  def impurity: String = _impurity
  def setImpurity(value: String): DTParams = { _impurity = value; this }

  private[this] var _maxDepth: Int = 5
  def maxDepth: Int = _maxDepth
  def setMaxDepth(value: Int): DTParams = { _maxDepth = value; this }

  private[this] var _maxBins: Int = 32
  def maxBins: Int = _maxBins
  def setMaxBins(value: Int): DTParams = { _maxBins = value;this }

}

object DTParams{

  // Possible Impurity values
  val IMPURITY_VARIANCE = "variance"  // Regression problem
  val IMPURITY_ENTROPY  = "entropy"   // Classification problem
  val IMPURITY_GINI     = "gini"      // Classification problem

}
