package com.stratio.intelligence.automaticBenchmark.dataset

import org.apache.spark.sql.DataFrame

import scala.collection.immutable.IndexedSeq

case class Fold( number:Int, trainDf:DataFrame, testDf:DataFrame )

object Fold{

  def generateFolds( abmDataset:AbmDataset, nfolds: Int, seed: Long): Array[Fold] ={

    assert( nfolds>1, "Number of folds must be greater than 1" )

    val df = abmDataset.df

    // First we divide the dataframe in two according to the class which is a number 0.0 or 1.0
    val df_positive = df.where( abmDataset.labelColumn + " = 1.0")
    val df_negative = df.where( abmDataset.labelColumn + " = 0.0")

    // Generating folds
    val foldsPositive: Array[DataFrame] = df_positive.randomSplit( Array.fill(nfolds)(1), seed )
    val foldsNegative: Array[DataFrame] = df_negative.randomSplit( Array.fill(nfolds)(1), seed )

    // Generating training and test folds
    val folds: IndexedSeq[Fold] = (0 until nfolds).map( i =>{
      val trainDf = ( foldsPositive.drop(i) ++ foldsNegative.drop(i) ).reduce( (df1,df2) => df1.unionAll(df2) )
      val testDf  = foldsPositive(i).unionAll(foldsNegative(i))
      Fold(i,trainDf,testDf)
    })

    folds.toArray
  }

}