package com.stratio.intelligence.automaticBenchmark.models

import com.stratio.intelligence.automaticBenchmark.AutomaticBenchmarkMachineLogger
import com.stratio.intelligence.automaticBenchmark.dataset.{AbmDataset, Fold}
import com.stratio.intelligence.automaticBenchmark.results._
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame


abstract class BenchmarkModel {

  // Logger
  val logger = AutomaticBenchmarkMachineLogger

  // Model name
  val MODEL_NAME:String

  // Categorical features required pre-processing steps
  def categoricalAsIndex: Boolean
  def categoricalAsBinaryVector:Boolean

  // Parameters of the model
  protected var modelParameters:ModelParameters = _

  // Trained model
  protected var trainedModel: Any = _


  /** Sets the model parameters */
  def setParameters( modelParams:ModelParameters ): BenchmarkModel

  /** Transforms the input fold in order to get the correct data and format for the training/testing method */
  def adequateData( dataset:AbmDataset, fold:DataFrame ):Any

  /** Executes an iteration of the benchmark: train/test evaluation using a series of folds */
  def executeBenchmark( dataset:AbmDataset, iterNumber:Integer, folds:Array[Fold] ): Array[BenchmarkResult] ={

    logger.logInfo(s"${this.MODEL_NAME} => Executing benchmark for dataset ${dataset.fileName}: Iter=${iterNumber}")

    // For each fold ...
    val iterationResults: Array[BenchmarkResult] =
      folds.map( fold => {

        try {
          // Getting training and testing data
          logger.logInfo(s"\t· Pre-processing train/test data:")
          val trainData = adequateData(dataset, fold.testDf)
          val testData = adequateData(dataset, fold.trainDf)

          // Train model with training data
          logger.logInfo(s"\t· Training model:")
            trainData match {
              case trainDf: DataFrame => logger.logInfo(s"\t\t· Persisting Dataframe"); trainDf.persist()
              case trainRDD: RDD[Any] => logger.logInfo(s"\t\t· Persisting RDD"); trainRDD.persist()
              case _ => println("Error")
            }
            logger.logInfo(s"\t\t· Training...");
            val trainingTime = measureTrainingTime( train(dataset, trainData) )
            trainData match {
              case trainDf: DataFrame => logger.logInfo(s"\t\t· Unpersisting Dataframe"); trainDf.unpersist(true)
              case trainRDD: RDD[Any] => logger.logInfo(s"\t\t· Unpersisting RDD"); trainRDD.unpersist(true)
              case _ => println("Error")
            }

          // Testing model with testing data
          logger.logInfo(s"\t· Getting predictions:")
          val predictions: RDD[(Double, Double)] = predict(testData)

          // Measure the trained model performance
          logger.logInfo(s"\t· Getting performance metrics:")
          val metrics: AbmMetrics = getMetrics(predictions)

          logger.logDebug(metrics.getSummary())

          SuccessfulBenchmarkResult(dataset.fileName, iterNumber, fold, this, metrics, trainingTime)

        }catch {
          case e:Exception => FailedBenchmarkResult(dataset.fileName, iterNumber, fold, this, e)
        }
      })

    iterationResults
  }

  def measureTrainingTime[A](f: => A): Double = {
    val s = System.nanoTime
    val ret = f
    val timeSeconds = (System.nanoTime-s)/1e9

    timeSeconds
  }

  def train[T]( dataset:AbmDataset, data:T )
  def predict[T]( data:T ):RDD[(Double,Double)]


  def getMetrics( predictionsAndLabels:RDD[(Double,Double)] ):AbmMetrics = {

    val abmBinaryClassificationMetrics = AbmBinaryClassificationMetrics()

    // => Multiclass Metrics
    val multiclassMetrics = new MulticlassMetrics(predictionsAndLabels)

      // · Confusion matrix
      val confusionMatrix = multiclassMetrics.confusionMatrix
        abmBinaryClassificationMetrics.confusionMatrixTP = confusionMatrix(1,1)
        abmBinaryClassificationMetrics.confusionMatrixTN = confusionMatrix(0,0)
        abmBinaryClassificationMetrics.confusionMatrixFP = confusionMatrix(0,1)
        abmBinaryClassificationMetrics.confusionMatrixFN = confusionMatrix(1,0)

      // · Overall Statistics
      abmBinaryClassificationMetrics.precision = multiclassMetrics.precision
      abmBinaryClassificationMetrics.recall    = multiclassMetrics.recall // same as true positive rate
      abmBinaryClassificationMetrics.f1Score   = multiclassMetrics.fMeasure

      // · Precision, Recall, FPR & F-measure by label
      val labels: Array[Double] = multiclassMetrics.labels
        abmBinaryClassificationMetrics.precisionLByLabel = labels.map(l => (l,multiclassMetrics.precision(l)) )
        abmBinaryClassificationMetrics.recalByLabel = labels.map(l => (l,multiclassMetrics.recall(l)) )
        abmBinaryClassificationMetrics.fprByLabel = labels.map(l => (l,multiclassMetrics.falsePositiveRate(l)) )
        abmBinaryClassificationMetrics.f1ScoreByLabel = labels.map(l => (l,multiclassMetrics.fMeasure(l)) )

      // · Weighted stats
      abmBinaryClassificationMetrics.weightedPrecision = multiclassMetrics.weightedPrecision
      abmBinaryClassificationMetrics.weightedRecall = multiclassMetrics.weightedRecall
      abmBinaryClassificationMetrics.weightedF1Score = multiclassMetrics.weightedFMeasure
      abmBinaryClassificationMetrics.weightedFalsePositiveRate = multiclassMetrics.weightedFalsePositiveRate

    // => Binary classification metrics, varying threshold
    val binayMetrics = new BinaryClassificationMetrics(predictionsAndLabels)

      // AUPRC
      abmBinaryClassificationMetrics.binaryAUPRC = binayMetrics.areaUnderPR

      // AUROC
      abmBinaryClassificationMetrics.binaryAUROC = binayMetrics.areaUnderROC

      // Precision by threshold
      val binaryPrecision: Array[(Double, Double)] = binayMetrics.precisionByThreshold.collect()
      abmBinaryClassificationMetrics.binaryPrecision = binaryPrecision

      // Recall by threshold
      abmBinaryClassificationMetrics.binaryRecall = binayMetrics.recallByThreshold.collect()

      // Precision-Recall Curve
      abmBinaryClassificationMetrics.binaryPRC = binayMetrics.pr.collect()

      // F-measure
      abmBinaryClassificationMetrics.binaryF1Score = binayMetrics.fMeasureByThreshold.collect()

      val beta = 0.5
      abmBinaryClassificationMetrics.binaryFScore = binayMetrics.fMeasureByThreshold(beta).collect()

      // Compute thresholds used in ROC and PR curves
      abmBinaryClassificationMetrics.thresholds= binaryPrecision.map(_._1)

      // ROC Curve
      val binaryRoc: RDD[(Double, Double)] = binayMetrics.roc
      abmBinaryClassificationMetrics.rocByThreshold = binaryRoc.collect()

    abmBinaryClassificationMetrics
  }

}
