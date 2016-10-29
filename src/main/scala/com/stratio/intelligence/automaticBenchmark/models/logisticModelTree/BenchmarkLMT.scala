package com.stratio.intelligence.automaticBenchmark.models.logisticModelTree

import com.stratio.intelligence.automaticBenchmark.AutomaticBenchmarkMachine
import com.stratio.intelligence.automaticBenchmark.dataset.AbmDataset
import com.stratio.intelligence.automaticBenchmark.models.{ModelParameters, BenchmarkModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.{LabeledPointLmt, LabeledPoint}
import org.apache.spark.mllib.tree.DecisionTreeLmt
import org.apache.spark.mllib.tree.model.DecisionTreeModelLmt
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class BenchmarkLMT extends BenchmarkModel{

  val PRUNING_TYPE_VALIDATION = "VALIDATION_PRUNING"  // stable variable : the name starts with capital. Do not change!
  val PRUNING_TYPE_FOLDS      = "FOLDS_PRUNING"
  val IMPURITY_ENTROPY        = "entropy"
  val IMPURITY_VARIANCE       = "variance"
  val IMPURITY_GINI           = "gini"

  override val MODEL_NAME: String = "Logistic Model Tree"

  var trainedModel: DecisionTreeModelLmt = _

  override def categoricalAsIndex: Boolean = true
  override def categoricalAsBinaryVector: Boolean = true

  override def setParameters(modelParams: ModelParameters): Unit = {

  }

  override def adecuateData(dataset: AbmDataset, fold: DataFrame): Any = {

    // Helper UDF -> Assures than a Vector column is a denseVector
    val toDenseVector = udf( (x:Vector) => x.toDense )

    // Selecting label, numeric features and oneHot categorical variables
    val label = dataset.labelColumn
    val featuresWithIndexedCategorical: Array[String] = dataset.numericalFeatures ++ dataset.indexedCategoricalFeatures
    val featuresWithOneHotCategorical:  Array[String] = dataset.numericalFeatures ++ dataset.oneHotCategoricalFeatures

    // Transforming dataframe with selected features and label to a RDD[LabeledPointLmt]

      // Vector assembler of features with indexed categorical
        val vAssemblerWithIndexedCategorical = new VectorAssembler()
            .setInputCols( featuresWithIndexedCategorical )
            .setOutputCol( "vectorizedFeaturesWithIndexedCategorical" )
      // Vector assembler of features with oneHot categorical
        val vAssemblerWithOneHotCategorical = new VectorAssembler()
          .setInputCols( featuresWithOneHotCategorical )
          .setOutputCol( "vectorizedFeaturesWithOneHotCategorical" )

    val rdd: RDD[LabeledPointLmt] =
      // All features columns in a new vector column. In this case, two vectors with indexedCat and OneHotCat
      vAssemblerWithOneHotCategorical.transform(
        vAssemblerWithIndexedCategorical.transform( fold )
      )
      // Assuring denseVector
      .withColumn( "denseVectorFeaturesWithIndexedCategorical",
                   toDenseVector(col("vectorizedFeaturesWithIndexedCategorical")) )
      .withColumn( "denseVectorFeaturesWithOneHotCategorical",
                   toDenseVector(col("vectorizedFeaturesWithOneHotCategorical")) )
      .map( row => {
        new LabeledPointLmt(
          row.getAs[Double](label),
          row.getAs[Vector]("denseVectorFeaturesWithIndexedCategorical"),
          row.getAs[Vector]("denseVectorFeaturesWithOneHotCategorical")
        )
      })

    rdd
  }


  override def train[T](dataset: AbmDataset, data: T): Unit = {

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
       DecisionTreeLmt.trainClassifierWithValidation(
        data.asInstanceOf[RDD[LabeledPointLmt]],
        2,
        categoricalFeaturesInfo,
        IMPURITY_GINI,
        10,
         20,
        -1,
        pruningRatio=0.1,
        prune = "AUC",
        minElements = 2000,
        seed=1,
        debugConsole=true
      )

  }

  override def predict[T](data: T): RDD[(Double, Double)] = {

    val model = this.trainedModel
    data match {
      case testRDD: RDD[LabeledPointLmt] => {
        testRDD.map {
          case LabeledPointLmt(label: Double, featWithIndexedCat: Vector, featWithOneHotCat: Vector) =>
            (label, model.predictWithLogisticFeatures(featWithIndexedCat, featWithOneHotCat))
        }
      }
      case _ => {
        println("Error")
        null
      }
    }
  }
}
