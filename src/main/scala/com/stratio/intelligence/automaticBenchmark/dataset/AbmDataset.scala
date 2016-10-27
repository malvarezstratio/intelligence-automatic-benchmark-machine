package com.stratio.intelligence.automaticBenchmark.dataset

import org.apache.spark.sql.DataFrame


case class AbmDataset(
  df: DataFrame,
  labelColumn: String,
  positiveLabelValue: String,
  categoricalColumns: Array[String],
  hasCategoricalFeats:Boolean
)




