package com.stratio.intelligence.automaticBenchmark.dataset

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType


case class AbmDataset(){

  // DatasetPath
    var fileName:String = ""

  // Readed dataframe
    var df: DataFrame = _
    var dfSchema:StructType = _

  // Label
    var labelColumn: String = ""
    var positiveLabelValue: String = ""

  // Numerical featurs
    var numericalFeatures: Array[String] = Array[String]()

  // Categorical features
    // · Raw categorical features (encoded as strings)
    var categoricalFeatures: Array[String] = Array[String]()
    // · Indexed categorical features (encoded as doubles)
    var indexedCategoricalFeatures: Array[String] = Array[String]()
    // · Vectorized categorical features (encoded as binary vector, oneHot encoding)
    var oneHotCategoricalFeatures: Array[String] = Array[String]()

    // Dictionary with all the information about transformed categorical features
    var transformedCategoricalDict:Map[(String,String),Map[Double,String]] = Map[(String,String),Map[Double,String]]()

  def hasCategoricalFeats: Boolean = categoricalFeatures.length > 0

  def getSummary():String = {
    s""" ---------------------------------------------------------------------------
       |  Dataset: ${fileName}
       | --------------------------------------------------------------------------
       |
       |  · Label column: $labelColumn
       |    - Positive value: $positiveLabelValue
       |  . Numerical features:
       |      ${numericalFeatures.mkString(",")}
       |  . Categorical features:
       |      ${categoricalFeatures.mkString(",")}
       |
       |  - Number of samples: ${df.count()}
       |  - Samples by class: ${df.groupBy(labelColumn).count().collect().mkString(",")}
       |
     """.stripMargin
  }

}




