package com.stratio.intelligence.automaticBenchmark.functions

import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel}
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.tree.configuration.FeatureType
import org.apache.spark.mllib.tree.model.NodeLmt
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame


object AutomaticBenchmarkFunctions {

  /**
    * Encode categorical variables as numbers starting in 0
    *
    *   - createIndexersMap provides indexes to the categorical data transforming strings to numbers so that they can be codified as dummy variables later.
    *   - transformCategoricalDF transforms the categorical columns in the dataframe to indexed categories
    *
    * Both functions are defined inside to avoid using them from the outside of the function.
    */
  def indexRawCategoricalFeatures(
                                                unsortedCategoricalColumns: Array[String], df: DataFrame ):
  (DataFrame, Array[String], Map[String, Array[String]]) = {

    /*
    // createIndexersMap y transformCategoricalDF are defined inside to avoid using them from the outside of the function
    def createIndexersMap(unsortedCategoricalColumns: Array[String], df: DataFrame): Map[String, StringIndexerModel] = {

      val categoricalColumns = df.columns.filter(x => unsortedCategoricalColumns.contains(x))
      val indexers = categoricalColumns.map(x => (x, new StringIndexer().setInputCol(x).setOutputCol(x + "Index").fit(df)))
      val indexersMap = mutable.Map[String, StringIndexerModel]()
      for (i <- 0 to (indexers.size - 1)) {
        indexersMap.put(indexers(i)._1, indexers(i)._2)
      }
      return indexersMap.toMap
    }


    def transformCategoricalDF(
                                categoricalColumns: Array[String],
                                indexersMap: Map[String, StringIndexerModel],
                               df: DataFrame): DataFrame = {

      val num_categorical_columns = categoricalColumns.size
      var arrayTransformed = new Array[DataFrame](num_categorical_columns)
      for (i <- 0 to num_categorical_columns - 1) {
        i match {
          case 0 => {
            arrayTransformed(0) = indexersMap.get(categoricalColumns(0)).get.transform(df)
          }
          case _ => {
            arrayTransformed(i) = indexersMap.get(categoricalColumns(i)).get.transform(arrayTransformed(i - 1))
          }
        }
      }
      val newColumns = df.columns.map(x =>
        x match {
          case myname if categoricalColumns.contains(myname) => myname + "Index"
          case _ => x
        })
      val df2 = arrayTransformed(num_categorical_columns - 1).select(newColumns.head, newColumns.tail: _*)
      return df2
    }
    */

    // val columnsWithoutClass: Array[String] = df.columns.dropRight(1)

    // Getting categorical features
      val categoricalColumns: Array[String] = df.columns.filter(x => unsortedCategoricalColumns.contains(x))

    // Filling null values with 'na' string
      val noNullDf = df.na.fill("na",categoricalColumns)

    // Index categorical features through a Pipeline process
      // Construct an array of stages (each stage is a StringIndexer)
      val indexCategoricalFeatsStages: Array[StringIndexer] =
        categoricalColumns.map( categoricalColName =>
          new StringIndexer().setInputCol(categoricalColName).setOutputCol(categoricalColName + "Index")
        )
      // Construct a pipeline with the array of stages
      val indexCategoricalFeatsPipeline = new Pipeline()
      indexCategoricalFeatsPipeline.setStages( indexCategoricalFeatsStages )
      // Fit the pipeline to obtain a pipelineModel
      val indexCategoricalFeatsModel: PipelineModel = indexCategoricalFeatsPipeline.fit(noNullDf)
      // Use the pipelineModel to transform a dataframe
      val indexedDF: DataFrame = indexCategoricalFeatsModel.transform(noNullDf)
      // Dropping raw categorical columns (encoded as strings)
      val transformedDF: DataFrame = categoricalColumns.foldLeft(indexedDF)((df, colName) => df.drop(colName) )

    // New categorical columns names
      val newCategoricalColNames: Array[String] = indexCategoricalFeatsStages.map(_.getOutputCol)

    // Map with each categorical feature and its transformation
    val catFeatValuesMap: Map[String, Array[String]] =
      indexCategoricalFeatsModel.stages.map(
        x => {
          val stringIndexerModel = x.asInstanceOf[StringIndexerModel]
          val catFeatName        = stringIndexerModel.getInputCol
          val catFeatIndexedName = stringIndexerModel.getOutputCol
          (catFeatName, stringIndexerModel.labels.zipWithIndex.map{ case(value,i) => s"$catFeatIndexedName:$value:$i"} )
        }
      ).toMap


    /*
      val indexersMap = createIndexersMap(unsortedCategoricalColumns, noNullDf)
      // val categoricalColumns = df.columns.filter(x => unsortedCategoricalColumns.contains(x))
      // val transformedDF = transformCategoricalDF(categoricalColumns, indexersMap, df)

      val categoricalColumnsWithIndex = columnsWithoutClass.zipWithIndex.filter(x => unsortedCategoricalColumns.contains(x._1))
      val varsWithValues: Map[String, Array[String]] = indexersMap.map(x => (x._1, x._2.labels.map(y => x._1 + ":" + y))) // labels is sorted in the
      val varsWithNumberOfValues: Map[String, Int] = indexersMap.map(x => (x._1, x._2.labels.size)) // same order of the codes
      val allNames: Array[String] = columnsWithoutClass.flatMap(x => x match {
          case categoricalColumn if categoricalColumns.contains(x) => varsWithValues.get(x).get
          case _ => Array(x)
        })

      val allNumbersOfValues: Array[Int] = columnsWithoutClass.map(x => x match {
        case categoricalColumn if categoricalColumns.contains(x) => varsWithNumberOfValues.get(x).get
        case _ => 1
      })
      val categoricalFeaturesMap: Map[Int, Int] = categoricalColumnsWithIndex.map(x => (x._2, allNumbersOfValues(x._2))).toMap

      return (transformedDF, categoricalFeaturesMap, allNames, columnsWithoutClass)
    */

    (transformedDF, newCategoricalColNames, catFeatValuesMap)
  }

  /**
    * Transform categorical variables already encoded as numeric into binary dummy variables
    *
    * We assume that the categorical features have been encoded as Doubles starting from 0.0, i. e., an entry of
    * categoricalFeaturesInfo of the form "2 -> 5" means that the feature with index 2 (the third feature) can take
    * values in {0.0, 1.0, 2.0, 3.0, 4.0}
    *
    * NOTE: the function below is NEVER used as it has been turned into an UDF in the following cell
    */

  def oneHotTransformer( indexedCatColNames:Array[String], df:DataFrame ): (DataFrame, Array[String]) = {

    // Encode as oneHot the categorical features through a Pipeline process
      // Construct an array of stages (each stage is a StringIndexer)
      val oneHotCatFeatsStages: Array[OneHotEncoder] =
        indexedCatColNames.map( categoricalColName =>
          new OneHotEncoder().setInputCol(categoricalColName).setOutputCol(categoricalColName + "OneHot")
        )
      // Construct a pipeline with the array of stages
      val indexCategoricalFeatsPipeline = new Pipeline()
      indexCategoricalFeatsPipeline.setStages( oneHotCatFeatsStages )
      // Fit the pipeline to obtain a pipelineModel
      val oneHotCatFeatsModel: PipelineModel = indexCategoricalFeatsPipeline.fit(df)
      // Use the pipelineModel to transform a dataframe
      val oneHotDF: DataFrame = oneHotCatFeatsModel.transform(df)
      // Dropping raw categorical columns (encoded as strings)
      val transformedDF: DataFrame = indexedCatColNames.foldLeft(oneHotDF)((df, colName) => df.drop(colName) )
    // New categorical columns
      val oneHotCatColNames: Array[String] = oneHotCatFeatsStages.map(_.getOutputCol)

    (transformedDF,oneHotCatColNames)
  }

  def nominalToNumericUDF(categoricalFeaturesInfo: Map[Int,Int]) = udf{
    (features:Vector) =>
      // Helper function: Binary operation for foldLeft
      def checkAndAppend(acc: (Array[Double], Int), value: Double): (Array[Double], Int) = {
        val (accArray, index) = acc
        val keys = categoricalFeaturesInfo.keySet
        val arrayToAppend: Array[Double] =
          if (keys contains index) {
            val numCategories = categoricalFeaturesInfo(index)
            assert(value.toInt < numCategories, "Categorical value bigger than numCategories")
            val array = Vectors.zeros(numCategories).toArray
            array.updated(value.toInt, 1.0)
          }
          else Array(value)
        (accArray ++ arrayToAppend, index + 1)
      }

      val initAcummulator = (Array[Double](), 0) // initial value of foldLeft accumulator
      Vectors.dense(
        features.toArray.foldLeft(initAcummulator)(checkAndAppend)._1
      )
  }

  /**
    * Create label-balanced folds in the dataframe to generate training-test pairs
    *
    * The fold assignment is placed in "m x n" aditional columns where "m" is the number of times the partition is
    * repeated and "n" is the number of folds in each partition
    */
  def makeAllFolds(df: DataFrame, classColumn: String, nfolds: Int, m: Int, seed: Long): Array[(DataFrame,DataFrame)] = {

    // First we divide the dataframe in two according to the class which is a number 0.0 or 1.0
    val df_positive = df.where(classColumn + " = 1.0")
    val df_negative = df.where(classColumn + " = 0.0")

    val npositives = df_positive.count.toInt
    val nnegatives = df_negative.count.toInt
    val allFolds: Array[(DataFrame, DataFrame)] = new Array[(DataFrame, DataFrame)](m * nfolds)

    for(k <- 1 to m ){

      val trainDFs : Array[DataFrame] = new Array[DataFrame](nfolds)
      val testDFs  : Array[DataFrame] = new Array[DataFrame](nfolds)
      val foldsDFs : Array[DataFrame] = new Array[DataFrame](nfolds)

      val foldsPositive = df_positive.randomSplit(Array.fill(nfolds)(1))
      val foldsNegative = df_negative.randomSplit(Array.fill(nfolds)(1))

      for(i <- 0 to (nfolds-1)){
        // A "fold" contains 1/nfolds positive + 1/nfolds negative examples
        foldsDFs(i) = foldsPositive(i).unionAll(foldsNegative(i))
      }

      val a: Unit = for(i <- 0 to (nfolds-1)){
        testDFs(i) = foldsDFs(i)
        for(j <- 0 to (nfolds-1)){
          if(i != j){
            if(trainDFs(i) == null){
              trainDFs(i) = foldsDFs(j)
            }
            else{
              trainDFs(i) = trainDFs(i).unionAll(foldsDFs(j))
            }
          }
        }
        allFolds(i + (k-1)*nfolds) = (testDFs(i), trainDFs(i))
      }

    }

    return allFolds
  }

  /**
    * Write the metrics to an output file specially formatted
    *
    * The function receives a multidimensional array and extracts the info to print it in the output file
    * in the specified format
    */

  def printByMetric(metricsArray:Array[Array[Array[Array[Any]]]], fileToWrite:String) ={

    import java.io._

    val writer = new PrintWriter(new File(fileToWrite))

    val metricNamesSingleValue = Array("confusionMatrixTP","confusionMatrixTN","confusionMatrixFP","confusionMatrixFN",
      "precision", "recall", "fMeasure", "Precision(0.0)", "Recall(0.0)", "FPR(0.0)",
      "F1-Score(0.0)", "Precision(1.0)", "Recall(1.0)", "FPR(1.0)", "F1-Score(1.0)",
      "AUPRC", "AUROC")
    var metric_counter = 0
    var dataset_counter = 0
    var fold_counter = 0
    var algorithm_counter = 0

    for (metric <- metricNamesSingleValue) {

      writer.write(metric)
      writer.write("\n")
      writer.write("\n")
      writer.write("\11")
      writer.write("\11")

      for (dataset <- metricsArray(0)) {

        dataset_counter = dataset_counter + 1
        fold_counter = 0

        for (fold <- dataset) {

          fold_counter = fold_counter + 1

          writer.write(s"DS_")
          writer.write(s"$dataset_counter \11")
          writer.write(s"Fold_")
          writer.write(s"$fold_counter \11")

          for (algorithm_counter <- 0 until metricsArray.length) {

            writer.write(metricsArray(algorithm_counter)(dataset_counter-1)(fold_counter-1)(metric_counter)
              .asInstanceOf[Double].toString)

          }
          writer.write("\n")
          writer.write("\11")
          writer.write("\11")
        }

      }

      metric_counter = metric_counter + 1
      dataset_counter = 0

      writer.write("\n")
      writer.write("\n")

    }
    writer.close()
  }

  /**
    * Printing the tree for LMT
    *
    *   This function receives the top node of an lmt model and prints the tree structure
    */

  def printTree(node: NodeLmt, selVars: Array[String]): String = {
    def printNode(node: NodeLmt, level: Int): String = {
      if (node.isLeaf) "RL" + node.id + "\n"
      else {
        val split = node.split.get
        val (splitDescriptionLeft, splitDescriptionRight) = {
          val feat = selVars(split.feature)
          split.featureType match {
            case FeatureType.Continuous => {
              val threshold = split.threshold
              (feat + " <= " + threshold, feat + " > " + threshold)
            }
            case FeatureType.Categorical => {
              val categories = "{" + split.categories.map(_.toInt).mkString(", ") + "}"
              (feat + " in " + categories, feat + " not in " + categories)
            }
          }
        }
        val (left, right) = (node.leftNode.get, node.rightNode.get)

        splitDescriptionLeft +
          {if (left.isLeaf) ": " else {"\n" + "|   " * level}} +
          printNode(left, level + 1) +
          "|   " * (level - 1) +
          splitDescriptionRight +
          {if (right.isLeaf) ": " else {"\n" + "|   " * level}} +
          printNode(right, level + 1)
      }
    }
    "\n" + printNode(node, 1)
  }




}

