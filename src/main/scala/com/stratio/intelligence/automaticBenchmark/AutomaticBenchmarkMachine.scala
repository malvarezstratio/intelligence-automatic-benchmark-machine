package com.stratio.intelligence.automaticBenchmark

import java.io.{File, PrintWriter}

import com.stratio.intelligence.automaticBenchmark.dataset.{AbmDataset, DatasetReader}
import com.stratio.intelligence.automaticBenchmark.functions.AutomaticBenchmarkFunctions
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

class AutomaticBenchmarkMachine(sqlContext: SQLContext) {

  // Logging and Debug
  private val logger = AutomaticBenchmarkMachineLogger
  def enableDebugMode()  = logger.DEBUGMODE = true
  def disableDebugMode() = logger.DEBUGMODE = false

  // SparkContext
  val sc = sqlContext.sparkContext

  /**
    *  Launches the automatic benchmarking process
    */
  def run(
           dataAndDescriptionFiles: Array[(String, String)],  // ("hdfs://data/mydata.csv", "hdfs://data/mydata.description")
           outputFile: String,                                // "hdfs://myoutput.txt"
           seed: Long,                                        // metrics: Array[String],
           kfolds: Integer,                                   // kfolds is the k for the k-fold CV
           mtimesFolds: Integer = 1,                          // mtimesFolds is the number of times to repeat the complete k-fold CV process independently
           // It should be replaced by: algorithms: Array[BenchmarkAlgorithm]
           algorithms: Array[String]
  ): Array[Array[Array[Array[Any]]]] = {

    // Getting sparkcontext from sqlContext
    val sc = sqlContext.sparkContext

    // New AutomaticBenchmarkFunctions
    val abmFuncs = AutomaticBenchmarkFunctions

    // New writer for output file
    val writer = new PrintWriter(new File(outputFile))

    // Parse the description file of each dataset, read the dataset, find the name of the class column, move it
    // to the right-most column, find out the positive label and whether there are categorical features or not
    val dataInfo: Array[AbmDataset] =  dataAndDescriptionFiles.map{
      case (datafile, descriptionfile) => DatasetReader.readDataAndDescription(sqlContext, datafile, descriptionfile)
    }

    // Find out which preprocessing steps are required for the set of algorithms to be run
    // so that each preprocessing is done only once
    var flagCatToDouble = false
    var flagCatToBinary = false

    // Replace the following with a call to methods getCatToDouble or getCatToBinary
    // in the class BenchmarkAlgorithm for each of the objects passed by the user
    algorithms.foreach { x => {
        x match {
          // The "match" should be replaced by
          //   flagCatToDouble = flagCatToDouble | x.getCatToDouble  (where x is an object of any subclass of BenchmarkAlgorithm)
          //   flagCatToBinary = flagCatToBinary | x.getCatToBinary
          case "LogisticRegression" => {
            flagCatToBinary = true // it is necessary to recode the categorical into binary features
          }
          case "LMT" => {
            flagCatToDouble = true // it is necessary to recode the categorical into numeric from 0 for DT
            flagCatToBinary = true // it is necessary to recode the categorical into binary features for Logistic
          }
          case "DecisionTree" => {
            flagCatToDouble = true // it is necessary to recode the categorical into numeric from 0
          }
        }
      }
    }

    // This is an empty DataFrame to be used in places where an actual DataFrame has no sense
    val emptySchema    = StructType(Array(StructField("dummy", StringType, true)))
    val emptyDataframe = sqlContext.createDataFrame(sc.emptyRDD[Row], emptySchema)


    val categoricalDFsWithInfo =
      dataInfo.map {
        // For each dataset
        case AbmDataset(df, classColumn, positiveLabel, categoricalColumns, categoricalPresent) => {

          // Dataframe which is going to pass to a transformation pipeline
          var transformedDf = df

          transformedDf.show()

          // Transform categorical features
          if (categoricalPresent && (flagCatToDouble || flagCatToBinary)) {

            // Index raw categorical features (encoded as strings, transformed to double index values starting at 0.0)
            val (catIndexedDf, newCatFeatColNames, catFeaturesValuesMap) =
              abmFuncs.indexRawCategoricalFeatures(categoricalColumns, transformedDf)
            transformedDf = catIndexedDf

            transformedDf.show()


            // OneHot encoding of categorical features
            transformedDf = if (flagCatToBinary) {
              val (oneHotDf:DataFrame, newCatOneHotFeatColNames) = abmFuncs.oneHotTransformer(newCatFeatColNames, transformedDf)
              oneHotDf
            } else {
              transformedDf
            }
          }

          transformedDf.show()

          // Combining all features in one vector column
          val features: Array[String] = df.drop(classColumn).columns
          transformedDf = new VectorAssembler()
            .setInputCols(features).setOutputCol("featuresVector").transform(transformedDf)
          //transformedDf = features.foldLeft(transformedDf)( (df,colname) => df.drop(colname))


          /*
            println("calling trainTestPairsDFs")
            val trainTestPairsDFs: Array[(DataFrame, DataFrame)] =
              abmFuncs.makeAllFolds(transformedDf, classColumn, kfolds, mtimesFolds, seed)

            trainTestPairsDFs.map { case (trainDF, testDF) =>

              var RDDcategTrain: RDD[LabeledPoint]  = sc.emptyRDD[LabeledPoint]
              var RDDcategTest: RDD[LabeledPoint]   = sc.emptyRDD[LabeledPoint]
              var RDDbinaryTrain: RDD[LabeledPoint] = sc.emptyRDD[LabeledPoint]
              var RDDbinaryTest: RDD[LabeledPoint]  = sc.emptyRDD[LabeledPoint]

              // This is done only once for each dataset. The resulting RDDs
              // can be passed directly to the train and predict methods of the algorithms,
              // except for the LMT which needs the finalDataFrame itself to create his own RDD[LabeledPointLmn]
              if (flagCatToDouble) {
                RDDcategTrain = trainDF.rdd.map(row => // "featuresCateg" should be in a public symbolic
                  new LabeledPoint(// constant of the abstract class BenchmarkAlgorithm
                    row.getAs[Double](classColumn),
                    row.getAs[Vector]("featuresCateg"))
                )

                RDDcategTest = testDF.rdd.map(row =>
                  new LabeledPoint(
                    row.getAs[Double](classColumn),
                    row.getAs[Vector]("featuresCateg"))
                )
              }

              if (flagCatToBinary) {
                RDDbinaryTrain = trainDF.rdd.map(row => // "featuresBinary" should be in a public symbolic
                  new LabeledPoint(// constant of the abstract class BenchmarkAlgorithm
                    row.getAs[Double](classColumn),
                    row.getAs[Vector]("featuresBinary"))
                )
                RDDbinaryTest = testDF.rdd.map(row =>
                  new LabeledPoint(
                    row.getAs[Double](classColumn),
                    row.getAs[Vector]("featuresBinary"))
                )
              }

              // The keys "fullDFtrain", "fullDFtest", etc must be defined in the abstract superclass as
              // symbolic string constants (public final) and accessed from here to assign the keys of RDDmap
              // e.g: val KEY_FULLDF_TRAIN = "fullDFtrain", etc

              val miscelaneaMap = mutable.HashMap(
                "fullDFtrain" -> trainDF,
                "fullDFtest"  -> testDF,
                "RDDcategTrain"  -> RDDcategTrain,
                "RDDcategTest"   -> RDDcategTest,
                "RDDbinaryTrain" -> RDDbinaryTrain,
                "RDDbinaryTest"  -> RDDbinaryTest,
                "categoricalFeaturesMap" -> categoricalFeaturesMap,
                "classColumn" -> classColumn
              )

              algorithms.map(algorithm => {
                if (algorithm == "LogisticRegression") {

                  val model = new BenchmarkLogisticRegression(sc, 2)
                  println("Training model ... \n")
                  val trainedModel = model.train(miscelaneaMap)
                  println("Model trained, predicting with test fold ... \n")
                  val predictionsAndLabels = model.predict(trainedModel, miscelaneaMap)
                  println("Calculating metrics ... \n")
                  val metrics = model.getMetrics(trainedModel, predictionsAndLabels)
                  println(metrics(15))
                  println(metrics(16))
                  println("\n")
                  /* AUC & AUPR
                  writer.write(metrics(15)._1)
                  writer.write("=")
                  writer.write(metrics(15)._2.toString)
                  writer.write("\n")
                  writer.write(metrics(16)._1)
                  writer.write("=")
                  writer.write(metrics(16)._2.toString)
                  writer.write("\n")
                  */

                } else if (algorithm == "DecisionTree") {
                  val model = new BenchmarkDecisionTree(sqlContext, m_impurity = "gini", m_maxDepth = 5, m_maxBins = 32, numClasses = 2)
                  println("Training model ... \n")
                  val trainedModel = model.train(miscelaneaMap)
                  println("Model trained, predicting with test fold... \n")
                  val predictionsAndLabels = model.predict(trainedModel, miscelaneaMap)
                  println("Calculating metrics... \n")
                  val metrics = model.getMetrics(trainedModel, predictionsAndLabels)
                  println(metrics(15))
                  println(metrics(16))
                  println("\n")
                  /* AUC & AUPR
                  writer.write(metrics(15)._1)
                  writer.write("=")
                  writer.write(metrics(15)._2.toString)
                  writer.write("\n")
                  writer.write(metrics(16)._1)
                  writer.write("=")
                  writer.write(metrics(16)._2.toString)
                  writer.write("\n")
                  */

                } else if (algorithm == "LMT") {

                  val model = new BenchmarkLMT(
                    sqlContext,
                    m_pruningType = "VALIDATION_PRUNING",
                    m_impurity = "gini",
                    m_maxDepth = 5,
                    m_maxBins = 32,
                    numClasses = 2,
                    m_numFolds = 5
                  )

                  println("Training model ... \n")
                    val trainedModel = model.train(miscelaneaMap)
                  println("Model trained, predicting with test fold... \n")
                    val predictionsAndLabels = model.predict(trainedModel, miscelaneaMap)
                  println("Calculating metrics... \n")
                    val metrics = model.getMetrics(trainedModel, predictionsAndLabels)
                  println(metrics(15))
                  println(metrics(16))
                  println("\n")
                  /* AUC & AUPR
                  writer.write(metrics(15)._1)
                  writer.write("=")
                  writer.write(metrics(15)._2.toString)
                  writer.write("\n")
                  writer.write(metrics(16)._1)
                  writer.write("=")
                  writer.write(metrics(16)._2.toString)
                  writer.write("\n")
                  */

                } else {
                  println("no model")
                }
              })
            }
          */
          transformedDf
        }
      } // End of iterating over datasets

    // categoricalDFsWithInfo is an Array of 4-tuples like:
    // (aDataFrame, categoricalFeaturesMap, Array[String], Array[String])
    // The last two elements are basically irrelevant as they contain information used just when printing the LMT
    // The second is a map with elements like [2 -> 5] saying that column index 2 has 5 different categorical values
    // The first is a large dataframe with either (a) the "compressed" categorical features (one column) as numeric, OR
    // (b) the "compressed" categorical variables (one column) AND the "compressed" binary variables (another column).
    // Case (a) occurs when none of the algorithms requires binary (dummy) features, so they are not calculated
    // Case (b) occurs when at least one of the algorithms requires binary features so they are appended to the DF
    val metricsArray = Array.ofDim[Any](2, 2, 2, 2)
    writer.close()

    metricsArray
  }
}

object AutomaticBenchmarkMachineLogger{

  var DEBUGMODE = false

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
