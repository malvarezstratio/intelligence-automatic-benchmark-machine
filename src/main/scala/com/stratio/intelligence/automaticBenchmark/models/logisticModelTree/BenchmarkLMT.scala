package com.stratio.intelligence.automaticBenchmark.models.logisticModelTree

import com.stratio.intelligence.automaticBenchmark.AutomaticBenchmarkMachine
import com.stratio.intelligence.automaticBenchmark.dataset.AbmDataset
import com.stratio.intelligence.automaticBenchmark.models.{BenchmarkModel, ModelParameters}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPointLmt
import org.apache.spark.mllib.tree.DecisionTreeLmt
import org.apache.spark.mllib.tree.model.DecisionTreeModelLmt
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class BenchmarkLMT extends BenchmarkModel{

  // Model name
  override val MODEL_NAME: String = "Logistic Model Tree"

  // Categorical features required pre-processing steps
  override def categoricalAsIndex: Boolean = true
  override def categoricalAsBinaryVector: Boolean = true

  // Parameters of the model
  modelParameters = LMTParams()

  // Categorical features map (required for training the model):
  //  · Map[ Index of the categorical feat. in the input features vector, Number of categories ]
  var categoricalFeaturesInfo:Map[Int,Int] = _

  /** Sets the model parameters */
  override def setParameters( modelParams: ModelParameters ): BenchmarkModel = {
    modelParams match {
      case m:LMTParams => this.modelParameters = m
      case _ => print("Error")
    }

    this
  }

  /** Transforms the input fold in order to get the correct data and format for the training/testing method */
  override def adequateData(dataset: AbmDataset, fold: DataFrame): Any = {

    // Helper UDF -> Assures than a Vector column is a denseVector
    val toDenseVector = udf( (x:Vector) => x.toDense )

    // Selecting label, numeric features and oneHot categorical variables
    val label = dataset.labelColumn
    val featuresWithIndexedCategorical: Array[String] = dataset.numericalFeatures ++ dataset.indexedCategoricalFeatures
    val featuresWithOneHotCategorical:  Array[String] = dataset.numericalFeatures ++ dataset.oneHotCategoricalFeatures

    // Transforming dataframe with selected features and label to a RDD[LabeledPointLmt]

      // · Vector assembler of features with indexed categorical
        val vAssemblerWithIndexedCategorical = new VectorAssembler()
            .setInputCols( featuresWithIndexedCategorical )
            .setOutputCol( "vectorizedFeatsWithIndexedCat" )
      // · Vector assembler of features with oneHot categorical
        val vAssemblerWithOneHotCategorical = new VectorAssembler()
          .setInputCols( featuresWithOneHotCategorical )
          .setOutputCol( "vectorizedFeatsWithOneHotCat" )

    val rdd: RDD[LabeledPointLmt] =
      // All features columns in a new vector column. In this case, two vectors with indexedCat and OneHotCat
      vAssemblerWithOneHotCategorical.transform(
        vAssemblerWithIndexedCategorical.transform( fold )
      )
      // Assuring denseVector
      .withColumn( "denseVectorFeatsWithIndexedCat",
                   toDenseVector(col("vectorizedFeatsWithIndexedCat")) )
      .withColumn( "denseVectorFeatsWithOneHotCat",
                   toDenseVector(col("vectorizedFeatsWithOneHotCat")) )
      .map( row => {
        new LabeledPointLmt(
          row.getAs[Double](label),
          row.getAs[Vector]("denseVectorFeatsWithIndexedCat"),
          row.getAs[Vector]("denseVectorFeatsWithOneHotCat")
        )
      })

    // Constructing indexed categorical features Map
      val numFeatures = featuresWithIndexedCategorical.length

      categoricalFeaturesInfo =
        (dataset.numericalFeatures.length until numFeatures).map( i => {
          val indexedCatFeat: String = featuresWithIndexedCategorical(i)
          val numberOfCategories: Int =
            dataset.transformedCategoricalDict.getOrElse(
              (indexedCatFeat.replaceAll(s"${AutomaticBenchmarkMachine.INDEXED_CAT_SUFFIX}$$",""),indexedCatFeat)
              , Map[Double,String]()
            ).size
          (i,numberOfCategories)
        }).toMap

    rdd
  }


  override def train[T](dataset: AbmDataset, data: T): Unit = {

    // TODO - Parameters of each case?? Training dependant of pruning strategy
    trainedModel =
      modelParameters.asInstanceOf[LMTParams].pruningType match {
        case LMTParams.PRUNING_TYPE_VALIDATION =>
          DecisionTreeLmt.trainClassifierWithValidation(
            data.asInstanceOf[RDD[LabeledPointLmt]],
            2,
            categoricalFeaturesInfo,
            modelParameters.asInstanceOf[LMTParams].impurity,
            modelParameters.asInstanceOf[LMTParams].maxDepth,
            modelParameters.asInstanceOf[LMTParams].maxBins,
            modelParameters.asInstanceOf[LMTParams].numLocalRegression,
            modelParameters.asInstanceOf[LMTParams].pruningRatio,
            modelParameters.asInstanceOf[LMTParams].weights,
            modelParameters.asInstanceOf[LMTParams].seed,
            modelParameters.asInstanceOf[LMTParams].costFunction,
            modelParameters.asInstanceOf[LMTParams].prune,
            modelParameters.asInstanceOf[LMTParams].numFolds,
            modelParameters.asInstanceOf[LMTParams].minElements,
            modelParameters.asInstanceOf[LMTParams].debugConsole
          )

        case LMTParams.PRUNING_TYPE_FOLDS =>
          DecisionTreeLmt.trainClassifierWithValidation(
            data.asInstanceOf[RDD[LabeledPointLmt]],
            2,
            categoricalFeaturesInfo,
            modelParameters.asInstanceOf[LMTParams].impurity,
            modelParameters.asInstanceOf[LMTParams].maxDepth,
            modelParameters.asInstanceOf[LMTParams].maxBins,
            modelParameters.asInstanceOf[LMTParams].numLocalRegression,
            modelParameters.asInstanceOf[LMTParams].pruningRatio,
            modelParameters.asInstanceOf[LMTParams].weights,//
            modelParameters.asInstanceOf[LMTParams].seed,
            modelParameters.asInstanceOf[LMTParams].costFunction,//
            modelParameters.asInstanceOf[LMTParams].prune,
            modelParameters.asInstanceOf[LMTParams].numFolds,//
            modelParameters.asInstanceOf[LMTParams].minElements,
            modelParameters.asInstanceOf[LMTParams].debugConsole
          )
      }
  }

  override def predict[T](data: T): RDD[(Double, Double)] = {

    val model = this.trainedModel.asInstanceOf[DecisionTreeModelLmt]
    val a =
      data.asInstanceOf[RDD[LabeledPointLmt]].map {
      case LabeledPointLmt(label: Double, featWithIndexedCat: Vector, featWithOneHotCat: Vector) =>
        (label, model.predictWithLogisticFeatures(featWithIndexedCat, featWithOneHotCat))    }

    a
  }
}
