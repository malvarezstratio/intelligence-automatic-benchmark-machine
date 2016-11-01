package com.stratio.intelligence.automaticBenchmark.models.logisticRegression

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters


case class LRParams() extends ModelParameters{

  // Property - Whether or not fit the interception term in the LR
  private[this] var _fitIntercept: Boolean = true
  def fitIntercept: Boolean = _fitIntercept
  def setFitIntercept(value: Boolean): LRParams = {
    _fitIntercept = value
    this
  }
}

object LRParams{

}