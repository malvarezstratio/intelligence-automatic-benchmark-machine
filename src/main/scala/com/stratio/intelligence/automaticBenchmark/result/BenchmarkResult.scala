package com.stratio.intelligence.automaticBenchmark.result

import com.stratio.intelligence.automaticBenchmark.dataset.Fold
import com.stratio.intelligence.automaticBenchmark.models.BenchmarkModel


case class BenchmarkResult(
  iteration:Int,
  fold:Fold,
  algorithm:BenchmarkModel,
  metrics:AbmMetrics
)

