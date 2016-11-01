package com.stratio.intelligence.automaticBenchmark.output

import java.io.{File, PrintWriter}

import com.stratio.intelligence.automaticBenchmark.AutomaticBenchmarkMachineLogger
import com.stratio.intelligence.automaticBenchmark.dataset.AbmDataset
import com.stratio.intelligence.automaticBenchmark.results.{SuccessfulBenchmarkResult, BenchmarkResult}


class OutputWriter( outputConf: OutputConf,
                    datasets:Array[AbmDataset],
                    benchmarkResults: Array[BenchmarkResult] ) {

  private val logger = AutomaticBenchmarkMachineLogger

  /** Saves the benchmarks results to a file */
  def saveSummaryToFile( ): Unit ={

    // New writer for output file
    val writer = new PrintWriter( new File(outputConf.filePath) )

    // Writing summaries for datasets and folds
    datasets.foreach( abmDataset => {
      writer.println(abmDataset.getSummary())
      abmDataset.folds.foreach( foldArray => foldArray.foreach( f => writer.println(f.getSummary())) )
    })

    // Writing benchmark results
    benchmarkResults.foreach( x =>{
      // Trained models
/*        x match {
          case x:SuccessfulBenchmarkResult =>
            writer.println(
              s""" Algorithm: ${x.algorithm.MODEL_NAME}
                 | Model: ${x.algorithm.getTrainedModelAsString(x.dataset,x.trainedModel)}
               """.stripMargin
            )
          case _ =>
        }*/
      // Metrics
      writer.println( x.getSummary( outputConf.showTrainedModel) )
    })


    writer.close()
  }


}
