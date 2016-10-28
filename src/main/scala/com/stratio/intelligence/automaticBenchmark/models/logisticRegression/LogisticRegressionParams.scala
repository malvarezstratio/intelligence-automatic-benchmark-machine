package com.stratio.intelligence.automaticBenchmark.models.logisticRegression

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters


case class LogisticRegressionParams() extends ModelParameters

object LogisticRegressionParams{

  // TODO
  def getDefaultParams():LogisticRegressionParams = {
    new LogisticRegressionParams()
  }
}