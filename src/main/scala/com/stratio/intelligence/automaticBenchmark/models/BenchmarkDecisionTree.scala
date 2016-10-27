package com.stratio.intelligence.automaticBenchmark.models

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Decision Tree
  *
  * - train method trains a Spark's Decision Tree Model from data received in a Labeled Point RDD and with the categorical values codified into dummy variables
  * - test method returns a prediction and label RDD with the predicted values as well as the true labels form a trained model and a test LabeledPoint RDD
  * - getMetrics method returns an array of metrics from an RDD with predictions and labels and the model itself, used to clear the threshold to return scores instead of labels
  *
  * NOTE: future versions should implement these methods into the main interface and pass a param to specify the algorithm to be tested
  */

class BenchmarkDecisionTree(sqlContext: SQLContext, m_impurity: String, m_maxDepth: Int, m_maxBins: Int, numClasses: Int) extends MLModel {

  val categoricalFeaturesInfo = Map[Int, Int]()
  val IMPURITY_ENTROPY = "entropy"
  val IMPURITY_VARIANCE = "variance"
  val IMPURITY_GINI = "gini"
  val sc = sqlContext.sparkContext

  def train(miscelaneaMap: scala.collection.mutable.HashMap[String, Object]): DecisionTreeModel = {

    val trainDF = miscelaneaMap.get(m_KEY_FULLDF_TRAIN).getOrElse {
      println("ERROR: BenchmarkDecisionTree.train in miscelaneaMap: None for key " + m_KEY_FULLDF_TRAIN + ". Creating empty DF.")
      //val emptySchema = StructType(StructField("k", StringType, true))
      sqlContext.createDataFrame(sc.emptyRDD[LabeledPoint]) // empty DataFrame
    }.asInstanceOf[DataFrame] // convert Any (as returned by the HashMap) to a specific type

    val categoricalFeaturesMap = miscelaneaMap.get(m_KEY_FEATURES_MAP).getOrElse {
      println("ERROR: BenchmarkDecisionTree.train in miscelaneaMap: None for key " + m_KEY_FEATURES_MAP + ". Creating empty Map")
      Map[Int, Int]()
    }.asInstanceOf[Map[Int, Int]]

    val trainRDD = miscelaneaMap.get(m_KEY_RDDCATEG_TRAIN).getOrElse {
      println("ERROR: BenchmarkLogisticRegression.train in miscelaneaMap: None for key " + m_KEY_RDDBINARY_TRAIN + ". Creating empty RDD.")
      //val emptySchema = StructType(StructField("k", StringType, true))
      sc.emptyRDD[LabeledPoint] // empty RDD
    }.asInstanceOf[RDD[LabeledPoint]] // convert Any (as returned by the HashMap) to a specific type

    val trainedModel = DecisionTree.trainClassifier(trainRDD, numClasses = numClasses, categoricalFeaturesMap, impurity = m_impurity,
      maxDepth = m_maxDepth, maxBins = m_maxBins)

    return trainedModel

  }

  def predict(trainedModel: DecisionTreeModel, miscelaneaMap: scala.collection.mutable.HashMap[String, Object]): RDD[(Double, Double)] = {

    val testRDD = miscelaneaMap.get(m_KEY_RDDCATEG_TEST).getOrElse {
      println("ERROR: BenchmarkLogisticRegression.train in miscelaneaMap: None for key " + m_KEY_RDDBINARY_TRAIN + ". Creating empty RDD.")

      sc.emptyRDD[LabeledPoint] // empty RDD
    }.asInstanceOf[RDD[LabeledPoint]] // convert Any (as returned by the HashMap) to a specific type

    val predictionAndLabels = testRDD.map { case LabeledPoint(label, features) =>
      val prediction = trainedModel.predict(features)
      (prediction, label)
    }

    return predictionAndLabels
  }

  /*override*/ def getMetrics(trainedModel: DecisionTreeModel, predictionsAndLabels: RDD[(Double, Double)]): List[(String, Any)] = {

    var metricsSummary: List[(String, Any)] = List()
    // Multiclass Metrics
    val multiclassMetrics = new MulticlassMetrics(predictionsAndLabels)

    val confusionMatrix = multiclassMetrics.confusionMatrix

    metricsSummary = metricsSummary :+("confusionMatrixTP", confusionMatrix(1, 1))
    metricsSummary = metricsSummary :+("confusionMatrixTN", confusionMatrix(0, 0))
    metricsSummary = metricsSummary :+("confusionMatrixFP", confusionMatrix(0, 1))
    metricsSummary = metricsSummary :+("confusionMatrixFN", confusionMatrix(1, 0))

    // Overall Statistics
    val precision = multiclassMetrics.precision
    metricsSummary = metricsSummary :+("precision", precision)

    val recall = multiclassMetrics.recall // same as true positive rate
    metricsSummary = metricsSummary :+("recall", recall)

    val f1Score = multiclassMetrics.fMeasure
    metricsSummary = metricsSummary :+("fMeasure", f1Score)

    // Precision, Recall, FPR & F-measure by label
    val labels = multiclassMetrics.labels
    labels.foreach { l =>
      metricsSummary = metricsSummary :+(s"Precision($l)", multiclassMetrics.precision(l))
      metricsSummary = metricsSummary :+(s"Recall($l)", multiclassMetrics.recall(l))
      metricsSummary = metricsSummary :+(s"FPR($l)", multiclassMetrics.falsePositiveRate(l))
      metricsSummary = metricsSummary :+(s"F1-Score($l)", multiclassMetrics.fMeasure(l))
    }

    // Weighted stats
    val weightedPrecision = multiclassMetrics.weightedPrecision
    val weightedRecall = multiclassMetrics.weightedRecall
    val weightedF1Score = multiclassMetrics.weightedFMeasure
    val weightedFalsePositiveRate = multiclassMetrics.weightedFalsePositiveRate

    // Binary classification metrics, varying  threshold
    //        trainedModel.clearThreshold decision tree does not have threshold

    val binayMetrics = new BinaryClassificationMetrics(predictionsAndLabels)

    // AUPRC
    val binaryAUPRC = binayMetrics.areaUnderPR
    metricsSummary = metricsSummary :+("AUPRC", binaryAUPRC)

    // AUROC
    val binaryAUROC = binayMetrics.areaUnderROC
    metricsSummary = metricsSummary :+("AUROC", binaryAUROC)

    // Precision by threshold
    val binaryPrecision = binayMetrics.precisionByThreshold
    metricsSummary = metricsSummary :+("precisionByThreshold", binaryPrecision.collect())

    // Recall by threshold
    val binaryRecall = binayMetrics.recallByThreshold
    metricsSummary = metricsSummary :+("recallByThreshold", binaryRecall.collect())

    // Precision-Recall Curve
    val binaryPRC = binayMetrics.pr
    metricsSummary = metricsSummary :+("PRCByThreshold", binaryPRC.collect())

    // F-measure
    val binaryF1Score = binayMetrics.fMeasureByThreshold
    metricsSummary = metricsSummary :+("F1ScoreByThreshold", binaryF1Score.collect())

    val beta = 0.5
    val binaryFScore = binayMetrics.fMeasureByThreshold(beta)
    metricsSummary = metricsSummary :+("FScoreByThreshold", binaryFScore.collect())

    // Compute thresholds used in ROC and PR curves
    val binaryThresholds = binaryPrecision.map(_._1)
    metricsSummary = metricsSummary :+("Thresholds", binaryThresholds.collect())

    // ROC Curve
    val binaryRoc = binayMetrics.roc
    metricsSummary = metricsSummary :+("ROCByThreshold", binaryRoc.collect())

    return metricsSummary

  }

}