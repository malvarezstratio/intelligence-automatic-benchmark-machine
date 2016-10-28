package com.stratio.intelligence.automaticBenchmark

import java.io.{File, PrintWriter}

import com.stratio.intelligence.automaticBenchmark.dataset.{Fold, AbmDataset, DatasetReader}
import com.stratio.intelligence.automaticBenchmark.functions.AutomaticBenchmarkFunctions
import com.stratio.intelligence.automaticBenchmark.models.BenchmarkModel
import com.stratio.intelligence.automaticBenchmark.result.BenchmarkResult
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import scala.collection.immutable.IndexedSeq

class AutomaticBenchmarkMachine( sqlContext: SQLContext ){

  // Logging and Debug
  private val logger = AutomaticBenchmarkMachineLogger
  def enableDebugMode()  = logger.DEBUGMODE = true
  def disableDebugMode() = logger.DEBUGMODE = false

  // SparkContext
  val sc = sqlContext.sparkContext

  /** Launches the automatic benchmarking process */
  def run(
           dataAndDescriptionFiles: Array[(String, String)],  // ("hdfs://data/mydata.csv", "hdfs://data/mydata.description")
           outputFile: String,                                // "hdfs://myoutput.txt"
           seed: Long,                                        // metrics: Array[String],
           kfolds: Integer,                                   // kfolds is the k for the k-fold CV
           mtimesFolds: Integer = 1,                          // mtimesFolds is the number of times to repeat the complete k-fold CV process independently
           // It should be replaced by: algorithms: Array[BenchmarkAlgorithm]
           algorithms: Array[BenchmarkModel]
  ) = {

    // Getting sparkcontext from sqlContext
    val sc = sqlContext.sparkContext

    // New AutomaticBenchmarkFunctions
    val abmFuncs = AutomaticBenchmarkFunctions

    // New writer for output file
    val writer = new PrintWriter(new File(outputFile))

    // Parse the description file of each dataset, read the dataset, find the name of the class column, move it
    // to the right-most column, find out the positive label and whether there are categorical features or not
    val dataInfo: Array[AbmDataset] = dataAndDescriptionFiles.map{
      case (datafile, descriptionfile) => DatasetReader.readDataAndDescription(sqlContext, datafile, descriptionfile)
    }

    // Find out which preprocessing steps are required for the set of algorithms to be run
    // so that each preprocessing is done only once
    val (flagCatToDouble, flagCatToBinary) = getPreprocessingSteps( algorithms )

    // For each dataset
    val benchmarkResults: Array[BenchmarkResult] =
      dataInfo.flatMap{ abmDataset => {

        logger.logInfo( s"· Processing dataset: ${abmDataset.fileName}")

        logger.logDebug( "=> Original dataframe: " )
        logger.logDebug( abmDataset.df .show() )

        // => Preprocessing steps

        // <> Transform categorical features
          if (abmDataset.hasCategoricalFeats && (flagCatToDouble || flagCatToBinary)) {

            // · Index raw categorical features (encoded as strings, transformed to double index values starting at 0.0)
            abmFuncs.indexRawCategoricalFeatures( abmDataset )

            logger.logDebug( "=> Dataframe with categorical features indexed: ")
            logger.logDebug( abmDataset.df.show() )

            // · OneHot encoding of categorical features
            if (flagCatToBinary) {
              abmFuncs.oneHotTransformer( abmDataset )
              logger.logDebug( "=> Dataframe with categorical features encoded using one hot: ")
              logger.logDebug( abmDataset.df.show() )
            }
          }

        // => Iterations
        val benchmarkResults: Array[BenchmarkResult] =
          ( 1 to mtimesFolds).flatMap( nIter => {

              // => Getting folds
                val folds: Array[Fold] = Fold.generateFolds( abmDataset, kfolds, seed )

              // => Iterating throught algorithms
                val modelResult: Array[BenchmarkResult] = algorithms.flatMap( model =>
                  model.executeBenchmark( abmDataset, nIter, folds )
                )
              modelResult
          }).toArray

        benchmarkResults
      }
    } // End of iterating over datasets

    println( benchmarkResults )

  }

  /**  Find out which preprocessing steps are required for the set of algorithms to be run
    *  so that each preprocessing is done only once
    */
  def getPreprocessingSteps( models:Array[BenchmarkModel] ): (Boolean,Boolean) ={

    // Find out which preprocessing steps are required for the set of algorithms to be run
    // so that each preprocessing is done only once
    models.foldLeft( (false,false) )( (catPrepro,model) =>{
      ( catPrepro._1 | model.categoricalAsIndex, catPrepro._2 | model.categoricalAsBinaryVector )
    })
  }

}

object AutomaticBenchmarkMachineLogger{

  var DEBUGMODE = false

  def logInfo(msn:String): Unit = println(msn)

  def logDebug(msn:String): Unit ={
    if(DEBUGMODE)
      println(msn)
  }

  def logDebug( op: => Unit ): Unit ={
    if(DEBUGMODE) {
      val stream = new java.io.ByteArrayOutputStream()
      Console.withOut(stream) { op }
      println(stream.toString)
      stream.close()
    }
  }
}
