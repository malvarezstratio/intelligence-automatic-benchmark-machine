package com.stratio.intelligence.automaticBenchmark.models.decisionTree

import com.stratio.intelligence.automaticBenchmark.AutomaticBenchmarkMachine
import com.stratio.intelligence.automaticBenchmark.dataset.AbmDataset
import com.stratio.intelligence.automaticBenchmark.models.{BenchmarkModel, ModelParameters}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


class BenchmarkDecisionTree extends BenchmarkModel{

  override val MODEL_NAME: String = "Decision tree"

  private var trainedModel: DecisionTreeModel = _
  modelParameters = DTParams()

  override def categoricalAsIndex: Boolean = true
  override def categoricalAsBinaryVector: Boolean = false

  override def setParameters(modelParams: ModelParameters): Unit = {
    modelParams match {
      case m:DTParams => this.modelParameters = m
      case _ => println("Error")
    }
  }

  override def adequateData(dataset: AbmDataset, fold: DataFrame ): Any = {

    // Selecting label, numeric features and indexed categorical variables
    val label = dataset.labelColumn
    val features: Array[String] = dataset.numericalFeatures ++ dataset.indexedCategoricalFeatures

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

  // TODO Mejorar el conteo de posibles categorias para cada variable categorica
  override def train[T]( dataset:AbmDataset, data: T): Unit = {

    val features: Array[String] = dataset.numericalFeatures ++ dataset.indexedCategoricalFeatures
    val numFeatures = features.length

    val categoricalFeaturesInfo: Map[Int, Int] =
      (dataset.numericalFeatures.length until numFeatures).map( i => {
        val indexedCatFeat: String = features(i)
        val numberOfCategories: Int =
          dataset.transformedCategoricalDict.getOrElse(
          (indexedCatFeat.replaceAll(s"${AutomaticBenchmarkMachine.INDEXED_CAT_SUFFIX}$$",""),indexedCatFeat)
          , Map[Double,String]()
        ).size
        (i,numberOfCategories)
      }).toMap

    trainedModel =
      DecisionTree.trainClassifier(
        data.asInstanceOf[RDD[LabeledPoint]],
        numClasses = 2,
        categoricalFeaturesInfo,
        impurity = modelParameters.asInstanceOf[DTParams].impurity,
        maxDepth = modelParameters.asInstanceOf[DTParams].maxDepth,
        maxBins  = modelParameters.asInstanceOf[DTParams].maxBins
      )
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

}
