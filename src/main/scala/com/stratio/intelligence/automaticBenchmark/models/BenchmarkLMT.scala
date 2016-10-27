package com.stratio.intelligence.automaticBenchmark.models

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPointLmt
import org.apache.spark.mllib.tree.DecisionTreeLmt
import org.apache.spark.mllib.tree.model.DecisionTreeModelLmt
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * LMT
  *
  * - train method trains a Stratio's Logistic Model Tree from data received in a LabeledPointLmtRDD and with the data prepared for both Logistic Regression and the Decision Tree. It implies that for the logistic regression the categorical values must be transformed to dummy variables, which is not necessary for the construction of the Decision Tree
  * - test method returns a prediction and label RDD with the predicted values as well as the true labels form a trained model and a test LabeledPointLmt RDD
  * - getMetrics method returns an array of metrics from an RDD with predictions and labels and the model itself, used to clear the threshold to return scores instead of labels
  *
  * NOTE: future versions should implement these methods into the main interface and pass a param to specify the algorithm to be tested
  */

class BenchmarkLMT(sqlContext: SQLContext, m_pruningType: String, m_impurity: String, m_maxDepth: Int, m_maxBins: Int, m_numFolds: Int, numClasses: Int) extends MLModel {

  val PRUNING_TYPE_VALIDATION = "VALIDATION_PRUNING"
  // stable variable : the name starts with capital. Do not change!
  val PRUNING_TYPE_FOLDS = "FOLDS_PRUNING"
  val IMPURITY_ENTROPY = "entropy"
  val IMPURITY_VARIANCE = "variance"
  val IMPURITY_GINI = "gini"
  val sc = sqlContext.sparkContext


  def train(miscelaneaMap: scala.collection.mutable.HashMap[String, Object]): DecisionTreeModelLmt = {

    val trainDF = miscelaneaMap.get(m_KEY_FULLDF_TRAIN).getOrElse {
      println("ERROR: BenchmarkLMT.train in miscelaneaMap: None for key " + m_KEY_FULLDF_TRAIN + ". Creating empty DF.")

      sqlContext.createDataFrame(sc.emptyRDD[LabeledPointLmt]) // empty DataFrame
    }.asInstanceOf[DataFrame] // convert Any (as returned by the HashMap) to a specific type

    val categoricalFeaturesMap = miscelaneaMap.get(m_KEY_FEATURES_MAP).getOrElse {
      println("ERROR: BenchmarkLMT.train in miscelaneaMap: None for key " + m_KEY_FEATURES_MAP + ". Creating empty Map")
      Map[Int, Int]()
    }.asInstanceOf[Map[Int, Int]]

    val classColumn = miscelaneaMap.get("classColumn").getOrElse {
      println("ERROR: BenchmarkLMT.train in miscelaneaMap: None for key classColumn. Creating empty string.")
      ""
    }.asInstanceOf[String]
    val compressedCategoricalFeatures = m_COMPRESSED_CATEG_FEATURES
    val compressedBinaryFeatures = m_COMPRESSED_BINARY_FEATURES

    val trainRDD = trainDF.rdd.map { row => {
      new LabeledPointLmt(
        row.getAs[Double](classColumn),
        row.getAs[Vector](compressedCategoricalFeatures),
        row.getAs[Vector](compressedBinaryFeatures)
      )
    }
    }

    val trainedModel = m_pruningType match {
      case PRUNING_TYPE_VALIDATION => {
        DecisionTreeLmt.trainClassifierWithValidation(
          trainRDD, 2, categoricalFeaturesMap, m_impurity, m_maxDepth, m_maxBins, 1000)
        // what was the 1000 ?-> minimum elements to run scala's logistic
      }
      case PRUNING_TYPE_FOLDS => {
        DecisionTreeLmt.trainClassifierWithValidation(
          trainRDD, numClasses, categoricalFeaturesMap,
          m_impurity, m_maxDepth, m_maxBins, -1, pruningRatio = 0.1, prune = "AUC",
          minElements = 2000, seed = 1, debugConsole = true)
        // what was the 1000 ?-> minimum elements to run scala's logistic
      }
    }
    //print(printTree(trainedModel.topNode, compressedCategoricalFeatures.split(" ")))
    //println(compressedCategoricalFeatures)
    println(m_COMPRESSED_BINARY_FEATURES)
    return trainedModel
  }

  def predict(predictionModel: DecisionTreeModelLmt, miscelaneaMap: scala.collection.mutable.HashMap[String, Object]):
  RDD[(Double, Double)] = {

    val testDF = miscelaneaMap.get(m_KEY_FULLDF_TEST).getOrElse {
      println("ERROR: BenchmarkLMT.train in miscelaneaMap: None for key " + m_KEY_FULLDF_TEST + ". Creating empty DF.")
      sqlContext.createDataFrame(sc.emptyRDD[LabeledPointLmt]) // empty DataFrame
    }.asInstanceOf[DataFrame]

    val categoricalFeaturesMap = miscelaneaMap.get(m_KEY_FEATURES_MAP).getOrElse {
      println("ERROR: BenchmarkLMT.train in miscelaneaMap: None for key " + m_KEY_FEATURES_MAP + ". Creating empty Map")
      Map[Int, Int]()
    }.asInstanceOf[Map[Int, Int]]

    val classColumn = miscelaneaMap.get("classColumn").getOrElse {
      println("ERROR: BenchmarkLMT.train in miscelaneaMap: None for key classColumn. Creating empty string.")
      ""
    }.asInstanceOf[String]
    val compressedCategoricalFeatures = m_COMPRESSED_CATEG_FEATURES
    val compressedBinaryFeatures = m_COMPRESSED_BINARY_FEATURES

    val testRDD = testDF.rdd.map { row => {
      new LabeledPointLmt(
        row.getAs[Double](classColumn),
        row.getAs[Vector](compressedCategoricalFeatures),
        row.getAs[Vector](compressedBinaryFeatures)
      )
    }
    }

    val predictionAndLabels = testRDD.map {
      case LabeledPointLmt(label, categFeatures, logisticFeatures) =>
        val prediction = predictionModel.predictWithLogisticFeatures(categFeatures, logisticFeatures)
        // <-- REVISAR (¿ NO NECESITA LAS BINARIAS TAMBIÉN ?)
        (prediction, label)
    }

    return predictionAndLabels

  }

  /*override*/ def getMetrics(trainedModel: DecisionTreeModelLmt, predictionsAndLabels: RDD[(Double, Double)]): List[(String, Any)] = {

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

    // Precision, Recall, FPR, F-measure by label
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
    //        trainedModel.clearThreshold LMT does not have threshold

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

