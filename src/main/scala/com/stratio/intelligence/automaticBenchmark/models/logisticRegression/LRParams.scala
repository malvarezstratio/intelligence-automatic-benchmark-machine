package com.stratio.intelligence.automaticBenchmark.models.logisticRegression

import com.stratio.intelligence.automaticBenchmark.models.ModelParameters


case class LRParams() extends ModelParameters{
  var fitIntercept:Boolean = true
}

object LRParams{

}