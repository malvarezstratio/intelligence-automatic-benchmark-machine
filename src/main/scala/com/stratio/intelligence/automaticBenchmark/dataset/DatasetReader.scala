package com.stratio.intelligence.automaticBenchmark.dataset

import com.stratio.intelligence.automaticBenchmark.{AutomaticBenchmarkMachineLogger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.types._

object DatasetReader {

  private val logger = AutomaticBenchmarkMachineLogger

  def readDataAndDescription( sqlContext: SQLContext, datafile:String, descriptionFile:String ): AbmDataset ={

      val abmDataset: AbmDataset = AbmDataset()

      // Parsing description file
      parseDescriptionFile( sqlContext, descriptionFile, abmDataset )

      // Reading data file
      abmDataset.df = sqlContext.read.format( "com.databricks.spark.csv" )
        .option("header", "false")
        .schema(abmDataset.dfSchema)
        .load(datafile)

      logger.logDebug( s"=> Readed data '$datafile': " )
      logger.logDebug( abmDataset.df.show() )

      // Getting categorical columns
      val categoricalColumns: Array[String] = abmDataset.df.schema.fields
        .filter(
          mystructfield =>
            (mystructfield.dataType == StringType) && (mystructfield.name != abmDataset.labelColumn ) )
        .map(_.name)

      // Getting numerical columns
      val numericalColumns: Array[String] = abmDataset.df.schema.fields
        .filter( field => (field.dataType == DoubleType) && (field.name != abmDataset.labelColumn ) )
        .map(_.name)

      // Find out whether the class column is a String field. In that case, it must be recoded as double
      val flagClassNeedsEncoding = abmDataset.df.schema.fields.exists(
        mystructfield => (mystructfield.name == abmDataset.labelColumn) && (mystructfield.dataType == StringType)
      ) // if it is NOT empty, then the class column has String type and must be recoded

      if (flagClassNeedsEncoding)
        encodeClassColumn( abmDataset )

      abmDataset.categoricalFeatures = categoricalColumns
      abmDataset.numericalFeatures = numericalColumns

      abmDataset
  }

  /**
    * Receives the path of a description file in csv format and returns the name of the class column,
    * the label that corresponds to the positive class, and the schema structure that must be used when
    * reading the dataset
    */
  def parseDescriptionFile( sqlContext:SQLContext, file: String, abmDataset: AbmDataset ): AbmDataset = {

    // Schema of the description file: csv with two columns
      val dictionarySchema = StructType(Array(
        StructField("colName", StringType, true),
        StructField("type", StringType, true)
      ))

    // Reading description file
      val descriptionDf =
        sqlContext.read.format("com.databricks.spark.csv")
          .option("header", "false")
          .schema(dictionarySchema).load(file)

      logger.logDebug( s"=> Readed description file '$file': " )
      logger.logDebug( descriptionDf.show() )

    // Getting label column and it's positive label
      abmDataset.labelColumn        = descriptionDf.head(1)(0)(1).toString // 2nd col of the first row:  class column
      abmDataset.positiveLabelValue = descriptionDf.head(2)(1)(1).toString // 2nd col of the second row: positive label

    // Reading dataset schema
      val myrddStructField: RDD[StructField] =
        descriptionDf.map {
          x =>
            if (x.getString(0) != "classcolumn" && x.getString(0) != "positivelabel") {
              val colname = x.getString(0)
              val coltype = x.getString(1).toLowerCase
              coltype match {
                case "double"  => StructField(colname, DoubleType)
                case "string"  => StructField(colname, StringType)
                case "integer" => StructField(colname, IntegerType)
              }
            } else {
              StructField("none", StringType)
            }
        }

      val myarray: Array[StructField] = myrddStructField.toArray.drop(2) // Drop the first two elements as it is a void StructField
      val mycustomSchema: StructType = StructType(myarray)

      abmDataset.dfSchema = mycustomSchema

    abmDataset
  }

  /** Recode the class column when it is a string, and convert it to doubles 1.0 or 0.0
    * The value "1.0", positive, corresponds to the positive class specified by the user via .description file
    */
  def encodeClassColumn( abmDataset: AbmDataset ) {

    val labelColumn = abmDataset.labelColumn
    val classType = abmDataset.df.schema.filter(_.name == labelColumn)(0).dataType

    if (classType == StringType) {
      // Value 1.0 should correspond to the user-specified positive label
      val classToNumeric =  udf { (make: String) => if( make == abmDataset.positiveLabelValue ) 1.0 else 0.0 }
      abmDataset.df =
        abmDataset.df
          .withColumn( labelColumn + "new", classToNumeric( col(labelColumn)) )
          .drop(labelColumn).withColumnRenamed( labelColumn + "new", labelColumn )
    }
  }

}
