package com.stratio.intelligence.automaticBenchmark.results

import com.stratio.intelligence.automaticBenchmark.dataset.{AbmDataset, Fold}
import com.stratio.intelligence.automaticBenchmark.models.BenchmarkModel

abstract class BenchmarkResult{
  def getSummary():String
}

case class SuccessfulBenchmarkResult (
                                       dataset:AbmDataset,
                                       iteration:Int,
                                       fold:Fold,
                                       algorithm:BenchmarkModel,
                                       trainedModel:Any,
                                       metrics:AbmMetrics,
                                       trainingTime:Double
) extends BenchmarkResult{

  def getSummary():String = {
    s""" Benchmark summary => State: SUCCESSFUL
       | ----------------------------------------------------------
       |    . Dataset: ${dataset.fileName}
       |    · Algorithm: ${algorithm.MODEL_NAME}
       |    · Iteration: ${iteration}
       |    · Fold: ${fold.foldNumber}
       |    · Training time: ${trainingTime}
       |    · Metrics:
       |        ${metrics.getSummary().replaceAll("\n","\n\t\t")}
     """.stripMargin
  }
}

case class FailedBenchmarkResult (
     dataset:AbmDataset,
     iteration:Int,
     fold:Fold,
     algorithm:BenchmarkModel,
     exception: Exception
   ) extends BenchmarkResult{

  def getSummary():String = {
    s""" Benchmark summary => State: FAILED
        | ----------------------------------------------------------
        |    . Dataset: ${dataset.fileName}
        |    · Algorithm: ${algorithm.MODEL_NAME}
        |    · Iteration: ${iteration}
        |    · Fold: ${fold.foldNumber}
        |    · Exception:
        |       ${exception.getStackTraceString.replaceAll("\n","\n\t\t")}
     """.stripMargin
  }
}

