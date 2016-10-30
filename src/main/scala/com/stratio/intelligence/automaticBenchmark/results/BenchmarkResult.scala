package com.stratio.intelligence.automaticBenchmark.results

import com.stratio.intelligence.automaticBenchmark.dataset.Fold
import com.stratio.intelligence.automaticBenchmark.models.BenchmarkModel


case class BenchmarkResult(
  dataset:String,
  iteration:Int,
  fold:Fold,
  algorithm:BenchmarkModel,
  metrics:AbmMetrics,
  trainingTime:Double
){

  def getSummary():String = {
    s""" Benchmark summary:
       | ----------------------------------------------------------
       |    . Dataset: ${dataset}
       |    · Algorithm: ${algorithm.MODEL_NAME}
       |    · Iteration: ${iteration}
       |    · Fold: ${fold.number}
       |    · Training time: ${trainingTime}
       |    · Metrics:
       |      ${metrics.getSummary().replaceAll("\n","\n\t\t")}
     """.stripMargin
  }
}

