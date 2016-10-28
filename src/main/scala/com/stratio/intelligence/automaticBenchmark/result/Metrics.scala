package com.stratio.intelligence.automaticBenchmark.result

abstract class AbmMetrics{

  def getSummary():String
}

case class AbmBinaryClassificationMetrics() extends AbmMetrics{

  var confusionMatrixTP: Double = _
  var confusionMatrixTN: Double = _
  var confusionMatrixFP: Double = _
  var confusionMatrixFN: Double = _

  // Overall Statistics
  var precision: Double = _
  var recall: Double = _
  var f1Score: Double = _

  // Precision Recall FPR & F-measure by label
  var precisionLByLabel: Array[(Double,Double)] = _
  var recalByLabel: Array[(Double, Double)] = _
  var fprByLabel: Array[(Double, Double)] = _
  var f1ScoreByLabel: Array[(Double, Double)] = _

  // Weighted stats = _
  var weightedPrecision: Double = _
  var weightedRecall: Double = _
  var weightedF1Score: Double = _
  var weightedFalsePositiveRate: Double = _

  // AUPRC
  var binaryAUPRC: Double = _

  // AUROC
  var binaryAUROC: Double = _

  // Precision by threshold
  var binaryPrecision: Array[(Double, Double)]= _

  // Recall by threshold
  var binaryRecall: Array[(Double, Double)] = _

  // Precision-Recall Curve
  var binaryPRC: Array[(Double, Double)] = _

  // F-measure
  var binaryF1Score: Array[(Double, Double)] = _

  var binaryFScore: Array[(Double, Double)] = _

  // Compute thresholds used in ROC and PR curves
  var thresholds: Array[Double] = _

  // ROC Curve
  var rocByThreshold: Array[(Double, Double)] = _


  // TODO
  override def getSummary(): String = {
    ""
  }

}

