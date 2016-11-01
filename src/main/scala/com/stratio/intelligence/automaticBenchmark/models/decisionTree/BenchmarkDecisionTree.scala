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
import org.apache.spark.sql.{UserDefinedFunction, DataFrame}
import org.apache.spark.sql.functions._


class BenchmarkDecisionTree extends BenchmarkModel{

  // Model name
  override val MODEL_NAME: String = "Decision tree"

  // Categorical features required pre-processing steps
  override def categoricalAsIndex: Boolean = true
  override def categoricalAsBinaryVector: Boolean = false

  // Parameters of the model
  modelParameters = DTParams()

  // Categorical features map (required for training the model):
  //  · Map[ Index of the categorical feat. in the input features vector, Number of categories ]
  var categoricalFeaturesInfo:Map[Int,Int] = _

  /** Sets the model parameters */
  override def setParameters(modelParams: ModelParameters):BenchmarkModel  = {
    modelParams match {
      case m:DTParams => this.modelParameters = m
      case _ => println("Error")
    }

    this
  }

  /** Transforms the input fold in order to get the correct data and format for the training/testing method */
  override def adequateData(dataset: AbmDataset, fold: DataFrame ): Any = {

    // Selecting label, numeric features and indexed categorical variables
    val label = dataset.labelColumn
    val features: Array[String] = dataset.numericalFeatures ++ dataset.indexedCategoricalFeatures

    // Transforming dataframe with selected features and label to a RDD[LabeledPoint]
    val toDenseVector: UserDefinedFunction = udf((x:Vector) => x.toDense )
    val vAssembler = new VectorAssembler().setInputCols(features).setOutputCol("vectorizedFeatures")

    val rdd: RDD[LabeledPoint] = vAssembler.transform( fold )
      .withColumn("denseVectorFeatures",toDenseVector(col("vectorizedFeatures")))
      .map( row => {
        new LabeledPoint(row.getAs[Double](label),row.getAs[Vector]("denseVectorFeatures") )
      })

    // Constructing indexed categorical features Map
    val numFeatures = features.length

    categoricalFeaturesInfo =
      (dataset.numericalFeatures.length until numFeatures).map( i => {
        val indexedCatFeat: String = features(i)
        val numberOfCategories: Int =
          dataset.transformedCategoricalDict.getOrElse(
            (indexedCatFeat.replaceAll(s"${AutomaticBenchmarkMachine.INDEXED_CAT_SUFFIX}$$",""),indexedCatFeat)
            , Map[Double,String]()
          ).size
        (i,numberOfCategories)
      }).toMap

    rdd
  }


  override def train[T]( dataset:AbmDataset, data: T): Unit = {

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

    val model = this.trainedModel.asInstanceOf[DecisionTreeModel]
    data.asInstanceOf[RDD[LabeledPoint]].map{
      case LabeledPoint(label:Double, features:Vector) => (label, model.predict(features)
    )}
  }

  override def getTrainedModelAsString(dataset:AbmDataset,model: Any):String = {
    model match {
      case m:DecisionTreeModel =>
        m.toDebugString
      case _ => "Error"
    }
  }

}
