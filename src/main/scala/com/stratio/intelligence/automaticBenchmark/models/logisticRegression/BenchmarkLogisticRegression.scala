package com.stratio.intelligence.automaticBenchmark.models.logisticRegression

import com.stratio.intelligence.automaticBenchmark.dataset.AbmDataset
import com.stratio.intelligence.automaticBenchmark.models.{BenchmarkModel, ModelParameters}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class BenchmarkLogisticRegression extends BenchmarkModel{

  override val MODEL_NAME: String = "Logistic regression"

  // Categorical features pre-processing steps
  override  def categoricalAsIndex: Boolean = true
  override  def categoricalAsBinaryVector: Boolean = true

  // Parameters of the model
  modelParameters = LRParams()


  /** Sets the model parameters */
  override def setParameters( modelParams: ModelParameters ): BenchmarkModel = {
    modelParams match {
      case m:LRParams => this.modelParameters = modelParams
      case _ => println("Error")
    }

    this
  }

  /** Transforms the input fold in order to get the correct data and format for the training/testing method */
  override def adequateData( dataset:AbmDataset, fold:DataFrame ):RDD[LabeledPoint] = {

    // Selecting label, numeric features and oneHot categorical variables
    val label = dataset.labelColumn
    val features: Array[String] = dataset.numericalFeatures ++ dataset.oneHotCategoricalFeatures

    // Transforming dataframe with selected features and label to a RDD[LabeledPoint]
    val toDenseVector = udf( (x:Vector) => x.toDense )
    val vAssembler = new VectorAssembler().setInputCols(features).setOutputCol("vectorizedFeatures")

    val rdd: RDD[LabeledPoint] = vAssembler.transform( fold )
      .withColumn("denseVectorFeatures",toDenseVector(col("vectorizedFeatures")))
      .map( row => {
        new LabeledPoint(row.getAs[Double](label),row.getAs[Vector]("denseVectorFeatures") )
    })

    rdd
  }

  override def train[T]( dataset:AbmDataset, data: T ): Unit = {
    trainedModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run( data.asInstanceOf[RDD[LabeledPoint]] )
  }

  override def predict[T](data: T): RDD[(Double,Double)] = {
    val model: LogisticRegressionModel = this.trainedModel.asInstanceOf[LogisticRegressionModel]
    data.asInstanceOf[RDD[LabeledPoint]].map{
      case LabeledPoint(label:Double, features:Vector) => (label, model.predict(features)
    )}
  }

}