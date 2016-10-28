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
    var categoricalFeatures: Array[String] = Array[String]()
    var indexedCategoricalFeatures: Array[String] = Array[String]()
    var oneHotCategoricalFeatures: Array[String] = Array[String]()


  def hasCategoricalFeats: Boolean = categoricalFeatures.length > 0

  // TODO
  def getSummary():String = {
    ""
  }

}




