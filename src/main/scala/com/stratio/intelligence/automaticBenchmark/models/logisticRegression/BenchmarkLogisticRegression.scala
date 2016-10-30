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

  override  def categoricalAsIndex: Boolean = true
  override  def categoricalAsBinaryVector: Boolean = true

  private var trainedModel: LogisticRegressionModel = _

  /** Sets the model parameters */
  override def setParameters( modelParams: ModelParameters ): Unit = {
    modelParams match {
      case m:LRParams => this.modelParameters = modelParams
      case _ => println("Error")
    }
  }

  override def adequateData(dataset:AbmDataset, fold:DataFrame ):RDD[LabeledPoint] = {

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

  // TODO - Warning match RDD[LabeledPoint]
  override def train[T]( dataset:AbmDataset, data: T ): Unit = {
    data match {
      case trainRDD:RDD[LabeledPoint] => {
         trainedModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run( trainRDD )
      }
      case _ => println("Error")
    }
  }

  override def predict[T](data: T): RDD[(Double,Double)] = {

    val model = this.trainedModel
    data match {
      case testRDD:RDD[LabeledPoint] => {
        testRDD.map{ case LabeledPoint(label:Double, features:Vector) => (label, model.predict(features) )}
      }
      case _ =>{
        println("Error")
        null
      }
    }
  }



  /*
  def train(miscelaneaMap: scala.collection.mutable.HashMap[String,Object]): LogisticRegressionModel ={

    val trainRDD = miscelaneaMap.get(m_KEY_RDDBINARY_TRAIN).getOrElse{
      println("ERROR: BenchmarkLogisticRegression.train in miscelaneaMap: None for key " + m_KEY_RDDBINARY_TRAIN
        + ". Creating empty RDD.")

      sc.emptyRDD[LabeledPoint] // empty RDD
    }.asInstanceOf[RDD[LabeledPoint]]  // convert Any (as returned by the HashMap) to a specific type

    val trainedModel = new LogisticRegressionWithLBFGS().setNumClasses(numClasses).run(trainRDD)

    return trainedModel

  }

  def predict(trainedModel: LogisticRegressionModel, miscelaneaMap: scala.collection.mutable.HashMap[String,Object]):
  RDD[(Double,Double)] = {

    val testRDD = miscelaneaMap.get(m_KEY_RDDBINARY_TEST).getOrElse{
      println("ERROR: BenchmarkLogisticRegression.predict in miscelaneaMap: None for key " + m_KEY_RDDBINARY_TEST
        + ". Creating empty RDD.")

      sc.emptyRDD[LabeledPoint] // empty RDD
    }.asInstanceOf[RDD[LabeledPoint]]  // convert Any (as returned by the HashMap) to a specific type
    trainedModel.clearThreshold
    val predictionAndLabels = testRDD.map { case LabeledPoint(label, features) =>
      val prediction = trainedModel.predict(features)
      (prediction, label)
    }
    print(predictionAndLabels.first())

    return predictionAndLabels
  }

  */

}