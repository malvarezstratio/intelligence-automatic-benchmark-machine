package com.stratio.intelligence.automaticBenchmark.models.decisionTree

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters

case class DTParams() extends ModelParameters{

  // · Impurity: Criterion used for information gain calculation.
  //      Classification -> Supported values: "gini" (recommended) or "entropy".
  // · MaxDepth: Maximum depth of the tree.
  //      E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (suggested value: 5)
  // · MaxBins: maximum number of bins used for splitting features (suggested value: 32)

  var impurity = DTParams.IMPURITY_GINI
  var maxDepth:Int = 5
  var maxBins:Int = 32

}

object DTParams{

  // Possible Impurity values
  val IMPURITY_VARIANCE = "variance"  // Regression problem
  val IMPURITY_ENTROPY  = "entropy"   // Classification problem
  val IMPURITY_GINI     = "gini"      // Classification problem

}
