package com.stratio.intelligence.automaticBenchmark.models

import com.stratio.intelligence.automaticBenchmark.dataset.{AbmDataset, Fold}
import com.stratio.intelligence.automaticBenchmark.results.{BenchmarkResult, AbmBinaryClassificationMetrics, AbmMetrics}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame


abstract class BenchmarkModel {

  protected var modelParameters:ModelParameters = _

  val MODEL_NAME:String

  def executeBenchmark( dataset:AbmDataset, iterNumber:Integer, folds:Array[Fold] ): Array[BenchmarkResult] ={

    val iterationResults: Array[BenchmarkResult] =
      folds.map( fold => {

        val trainData = adequateData( dataset, fold.testDf  )
        val testData  = adequateData( dataset, fold.trainDf )

        train( dataset,trainData )
        val predictions: RDD[(Double, Double)] = predict( testData )

        val metrics: AbmMetrics = getMetrics( predictions )

        BenchmarkResult( dataset.fileName,iterNumber, fold, this, metrics )
      })

    iterationResults
  }

  def categoricalAsIndex: Boolean
  def categoricalAsBinaryVector:Boolean

  def adequateData(dataset:AbmDataset, fold:DataFrame ):Any
  def setParameters( modelParams:ModelParameters )
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
