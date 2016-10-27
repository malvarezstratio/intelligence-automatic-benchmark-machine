package com.stratio.intelligence.automaticBenchmark.dataset

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.types._

object DatasetReader {

  def readDataAndDescription( sqlContext: SQLContext, datafile:String, descriptionFile:String ): AbmDataset ={

      // Parsing description file
      val (classColumn: String, positiveLabel: String, customSchema: StructType, categoricalPresent: Boolean) =
        parseDescriptionFile(sqlContext, descriptionFile)
      // Reading data file
      var df1 = sqlContext.read.format("com.databricks.spark.csv")
        .option("header", "false")
        .schema(customSchema).load(datafile)
      df1.show()

      // Getting categorical columns
      val categoricalColumns: Array[String] = df1.schema.fields
        .filter(
          mystructfield =>
            ( (mystructfield.dataType == StringType) && (mystructfield.name != classColumn)) )
        .map(_.name)

      // Find out whether the class column is a String field. In that case, it must be recoded as double
      val flagClassNeedsEncoding = !(df1.schema.fields.filter(mystructfield =>
        ((mystructfield.name == classColumn) && (mystructfield.dataType == StringType))
      ).isEmpty) // if it is NOT empty, then the class column has String type and must be recoded
      if (flagClassNeedsEncoding) {
        df1 = encodeClassColumn(df1, classColumn, positiveLabel)
      }

    AbmDataset(df1, classColumn, positiveLabel, categoricalColumns, categoricalPresent)
  }

  /**
    * Receives the path of a description file in csv format and returns the name of the class column,
    * the label that corresponds to the positive class, and the schema structure that must be used when
    * reading the dataset
    */
  def parseDescriptionFile(sqlContext:SQLContext, file: String): (String, String, StructType, Boolean) = {

    val dictionarySchema = StructType(Array(
      StructField("colName", StringType, true),
      StructField("type", StringType, true)
    ))
    val descriptionDf = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").
      schema(dictionarySchema).load(file)

    descriptionDf.show()

    var flagCategorical = false // whether there exist categorical columns in this dataset or not

    val myclassColumn = descriptionDf.head(1)(0)(1).toString // second column of the first row -> class column
    val mypositiveLabel = descriptionDf.head(2)(1)(1).toString // second column of the second row -> positive label
    val myrddStructField =
      descriptionDf.map {
        x =>
          if (x.getString(0) != "classcolumn" && x.getString(0) != "positivelabel") {
            val colname = x.getString(0)
            val coltype = x.getString(1).toLowerCase
            coltype match {
              case "double" => StructField(colname, DoubleType)
              case "string" => StructField(colname, StringType)
              case "integer" => StructField(colname, IntegerType)
            }
          } else {
            StructField("none", StringType)
          }
      }

    flagCategorical = !(myrddStructField.filter { x => x.dataType == StringType && x.name != "none" }.isEmpty)

    val myarray = myrddStructField.toArray.drop(2) // Drop the first two elements as it is a void StructField
    val mycustomSchema = org.apache.spark.sql.types.StructType(myarray)

    return (myclassColumn, mypositiveLabel, mycustomSchema, flagCategorical)
  }

  /** Recode the class column when it is a string, and convert it to doubles 1.0 or 0.0
    * The value "1.0", positive, corresponds to the positive class specified by the user via .description file
    */
  def encodeClassColumn(df: DataFrame, classColumn: String, PositiveLabel: String): DataFrame = {
    val classType = df.schema.filter(_.name == classColumn)(0).dataType

    val dffinal: DataFrame =
      if (classType == StringType) {
        // Value 1.0 should correspond to the user-specified positive label
        val classToNumeric =
          udf { (make: String) =>
            make match {
              case PositiveLabel => 1.0 // variable name PositiveLabel must start with capital P to work inside the match
              case _ => 0.0
            }
          }
        df.withColumn(classColumn, classToNumeric(df(classColumn)))
      } else {
        df
      }

    // Move the class column to the right-most position
    var dffinal2: DataFrame = dffinal.withColumn(classColumn + "new", dffinal(classColumn))
    dffinal2 = dffinal2.drop(classColumn)
    dffinal2 = dffinal2.withColumnRenamed(classColumn + "new", classColumn)

    return dffinal2
  }

}
