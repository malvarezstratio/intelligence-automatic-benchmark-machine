package com.stratio.intelligence.automaticBenchmark

import com.stratio.intelligence.automaticBenchmark.dataset.{AbmDataset, DatasetReader, Fold}
import com.stratio.intelligence.automaticBenchmark.models.BenchmarkModel
import com.stratio.intelligence.automaticBenchmark.output.{OutputConf, OutputWriter}
import com.stratio.intelligence.automaticBenchmark.results.BenchmarkResult
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}

class AutomaticBenchmarkMachine( sqlContext: SQLContext ){

  // Logging and Debug
  private val logger = {AutomaticBenchmarkMachineLogger.DEBUGMODE=false; AutomaticBenchmarkMachineLogger}
  def enableDebugMode()  = { logger.DEBUGMODE = true  }
  def disableDebugMode() = { logger.DEBUGMODE = false }

  // SparkContext
  val sc = sqlContext.sparkContext

  /** Launches the automatic benchmarking process */
  def run(
           dataAndDescriptionFiles: Array[(String, String)],  // ("hdfs://data/mydata.csv", "hdfs://data/mydata.description")
           outputConf: OutputConf,
           seed: Long,                                        // metrics: Array[String],
           kfolds: Integer,                                   // kfolds is the k for the k-fold CV
           mtimesFolds: Integer = 1,                          // mtimesFolds is the number of times to repeat the complete k-fold CV process independently
           algorithms: Array[BenchmarkModel]
  ): Array[BenchmarkResult] = {

    // => Parse the description file of each dataset, read the dataset,
    // find out the positive label and whether there are categorical features or not
    val datasets: Array[AbmDataset] = dataAndDescriptionFiles.map{
      case (datafile, descriptionfile) => DatasetReader.readDataAndDescription(sqlContext, datafile, descriptionfile)
    }

    // => Find out which pre-processing steps are required for the set of algorithms to be run
    // so that each pre-processing is done only once
    val (flagCatToDouble, flagCatToBinary) = getPreprocessingSteps( algorithms )

    // => For each dataset
    val benchmarkResults: Array[BenchmarkResult] =
      datasets.flatMap{ abmDataset => {

        logger.logInfo( s"· Processing dataset: ${abmDataset.fileName}")
        logger.logInfo( abmDataset.getSummary() )


        logger.logDebug( "=> Original dataframe: " )
        logger.logDebug( abmDataset.df .show() )

        // => Pre-processing steps

        // <> Transform categorical features
          if (abmDataset.hasCategoricalFeats && (flagCatToDouble || flagCatToBinary)) {

            // · Index raw categorical features (encoded as strings, transformed to double index values starting at 0.0)
              indexRawCategoricalFeatures( abmDataset )

            // · OneHot encoding of categorical features
              if (flagCatToBinary) oneHotTransformer( abmDataset )
          }

        // => Iterations
        abmDataset.df.cache()

        val benchmarkResults: Array[BenchmarkResult] =
          ( 1 to mtimesFolds).flatMap( nIter => {

              // => Getting folds
                val folds: Array[Fold] = Fold.generateFolds( abmDataset, nIter, kfolds, seed )

              // => Iterating throught algorithms
                val modelResult: Array[BenchmarkResult] = algorithms.flatMap( model =>
                  model.executeBenchmark( abmDataset, nIter, folds )
                )
              modelResult
          }).toArray

        abmDataset.df.unpersist()


        benchmarkResults
      }
    } // End of iterating over datasets


    // Writing output results
    val outputWriter = new OutputWriter(outputConf,datasets,benchmarkResults)
      // Summary to text file
      outputWriter.saveSummaryToFile()

    benchmarkResults
  }

  /**  Find out which pre-processing steps are required for the set of algorithms to be run
    *  so that each pre-processing is done only once
    */
  def getPreprocessingSteps( models:Array[BenchmarkModel] ): (Boolean,Boolean) ={

    models.foldLeft( (false,false) )( (catPrepro,model) =>{
      ( catPrepro._1 | model.categoricalAsIndex, catPrepro._2 | model.categoricalAsBinaryVector )
    })
  }

  /**
    * Encode categorical variables as numbers starting in 0
    *
    *   - createIndexersMap provides indexes to the categorical data transforming strings to numbers so that they can be codified as dummy variables later.
    *   - transformCategoricalDF transforms the categorical columns in the dataframe to indexed categories
    *
    * Both functions are defined inside to avoid using them from the outside of the function.
    */
  def indexRawCategoricalFeatures( abmDataset:AbmDataset ) = {

    val df = abmDataset.df

    val unsortedCategoricalColumns: Array[String] = abmDataset.categoricalFeatures

    // Getting categorical features
    val categoricalColumns: Array[String] = df.columns.filter(x => unsortedCategoricalColumns.contains(x))

    // Filling null values with 'na' string
    val noNullDf = df.na.fill("na",categoricalColumns)

    // Index categorical features through a Pipeline process
    // Construct an array of stages (each stage is a StringIndexer)
    val indexCategoricalFeatsStages: Array[StringIndexer] =
      categoricalColumns.map( categoricalColName =>
        new StringIndexer().setInputCol(categoricalColName)
          .setOutputCol( categoricalColName + AutomaticBenchmarkMachine.INDEXED_CAT_SUFFIX )
      )
    // Construct a pipeline with the array of stages
    val indexCategoricalFeatsPipeline = new Pipeline()
    indexCategoricalFeatsPipeline.setStages( indexCategoricalFeatsStages )
    // Fit the pipeline to obtain a pipelineModel
    val indexCategoricalFeatsModel: PipelineModel = indexCategoricalFeatsPipeline.fit(noNullDf)
    // Use the pipelineModel to transform a dataframe
    val transformedDF: DataFrame = indexCategoricalFeatsModel.transform(noNullDf)

    // New categorical columns names
    val newCategoricalColNames: Array[String] = indexCategoricalFeatsStages.map(_.getOutputCol)

    // Dictionary: map with each categorical feature and its transformation
    val catFeatValuesMap: Map[(String,String),Map[Double,String]]  =
      indexCategoricalFeatsModel.stages.map(
        x => {
          val stringIndexerModel = x.asInstanceOf[StringIndexerModel]
          val catFeatName        = stringIndexerModel.getInputCol
          val catFeatIndexedName = stringIndexerModel.getOutputCol

          ( (catFeatName,catFeatIndexedName), // (raw cat. colname, indexed cat. colname)
            stringIndexerModel.labels.zipWithIndex.map(x => (x._2.toDouble,x._1)).toMap
          )
        }
      ).toMap
    abmDataset.transformedCategoricalDict = catFeatValuesMap

    abmDataset.indexedCategoricalFeatures = newCategoricalColNames
    abmDataset.df = transformedDF

    logger.logDebug( "=> Dataframe with categorical features indexed: ")
    logger.logDebug( abmDataset.df.show() )
  }

  /**
    * Transform categorical variables already encoded as numeric into binary dummy variables
    *
    * We assume that the categorical features have been encoded as Doubles starting from 0.0, i. e., an entry of
    * categoricalFeaturesInfo of the form "2 -> 5" means that the feature with index 2 (the third feature) can take
    * values in {0.0, 1.0, 2.0, 3.0, 4.0}
    *
    * NOTE: the function below is NEVER used as it has been turned into an UDF in the following cell
    */

  def oneHotTransformer( abmDataset:AbmDataset ) = {

    val df = abmDataset.df
    val indexedCatColNames = abmDataset.indexedCategoricalFeatures

    // UDF - Assures DenseVector in a VectorType column
    val toDenseVector = udf( (x:Vector) => x.toDense )

    // Encode as oneHot the categorical features through a Pipeline process
    // Construct an array of stages (each stage is a StringIndexer)
    val oneHotCatFeatsStages: Array[OneHotEncoder] =
      indexedCatColNames.map( categoricalColName =>
        new OneHotEncoder()
          .setInputCol(categoricalColName)
          .setOutputCol(
            categoricalColName.replaceAll(
              s"${AutomaticBenchmarkMachine.INDEXED_CAT_SUFFIX}$$","") +
              s"${AutomaticBenchmarkMachine.ONEHOT_CAT_SUFFIX}Aux")
          .setDropLast(false)
      )
    // Construct a pipeline with the array of stages
    val indexCategoricalFeatsPipeline = new Pipeline()
      indexCategoricalFeatsPipeline.setStages( oneHotCatFeatsStages )
    // Fit the pipeline to obtain a pipelineModel
    val oneHotCatFeatsModel: PipelineModel = indexCategoricalFeatsPipeline.fit(df)
    // Use the pipelineModel to transform a dataframe
    val auxTransformedDF: DataFrame = oneHotCatFeatsModel.transform(df)

    // Assure dense vectors
    val transformedDF: DataFrame = oneHotCatFeatsStages.map(_.getOutputCol).foldLeft(auxTransformedDF)(
      (df,colname) => {
        df.withColumn(colname.replaceAll("Aux$",""),toDenseVector(col(colname))).drop(colname)
      }
    )

    // New categorical columns
    val oneHotCatColNames: Array[String] = oneHotCatFeatsStages.map(_.getOutputCol.replaceAll("Aux$",""))

    abmDataset.oneHotCategoricalFeatures = oneHotCatColNames
    abmDataset.df = transformedDF

    logger.logDebug( "=> Dataframe with categorical features encoded using one hot: ")
    logger.logDebug( abmDataset.df.show() )
  }
}

object AutomaticBenchmarkMachine{
  val INDEXED_CAT_SUFFIX = "_asIndex"
  val ONEHOT_CAT_SUFFIX  = "_asOneHot"
}

object AutomaticBenchmarkMachineLogger{

  var DEBUGMODE: Boolean = false

  def logInfo(msn:String): Unit = println( Console.YELLOW + msn + Console.RESET )

  def logDebug(msn:String): Unit ={
    if(DEBUGMODE)
      println( Console.RED + msn + Console.RESET )
  }

  def logDebug( op: => Unit ): Unit ={
    if(DEBUGMODE) {
      val stream = new java.io.ByteArrayOutputStream()
      Console.withOut(stream) { op }
      println( Console.RED + stream.toString + Console.RESET )
      stream.close()
    }
  }
}
