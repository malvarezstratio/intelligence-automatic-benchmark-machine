package com.stratio.intelligence.automaticBenchmark.dataset

import com.stratio.intelligence.automaticBenchmark.AutomaticBenchmarkMachineLogger
import org.apache.spark.sql.DataFrame

case class Fold( abmDataset: AbmDataset,
                 numIter:Int,
                 foldNumber:Int,
                 trainDf:DataFrame,
                 testDf:DataFrame         ){

  val nT   = trainDf.count()
  val nC1T = trainDf.where( abmDataset.labelColumn + " = 1.0" ).count()
  val nC0T = trainDf.where( abmDataset.labelColumn + " = 0.0" ).count()
  val nt   = testDf.count()
  val nC1t = testDf.where( abmDataset.labelColumn + " = 1.0" ).count()
  val nC0t = testDf.where( abmDataset.labelColumn + " = 0.0" ).count()


  def getSummary():String = {
    s""" =>Fold Summary
       |   · Dataset: ${abmDataset.fileName}
       |   · Iteration number: ${numIter} · Fold number: ${foldNumber}
       |   · Training data:  - N: ${nT} - N_C1: ${nC1T}  - N_C0: ${nC0T}
       |   · Testing data:   - N: ${nt} - N_C1: ${nC1t}  - N_C0: ${nC0t}
     """.stripMargin
  }
}

object Fold{

  val logger = AutomaticBenchmarkMachineLogger

  def generateFolds( abmDataset:AbmDataset, numIter:Int, nfolds: Int, seed: Long): Array[Fold] ={

    def dropEle(nth: Int, in: Array[DataFrame]): Array[DataFrame] = {
      in.view.zipWithIndex.filter{ _._2 != nth }.map{ _._1 }.toArray
    }

    assert( nfolds>1, "Number of folds must be greater than 1" )

    val df = abmDataset.df

    // First we divide the dataframe in two according to the class which is a number 0.0 or 1.0
    val df_positive = df.where( abmDataset.labelColumn + " = 1.0" )
    val df_negative = df.where( abmDataset.labelColumn + " = 0.0" )

    // Generating folds
    val foldsPositive: Array[DataFrame] = df_positive.randomSplit( Array.fill(nfolds)(1), seed )
    val foldsNegative: Array[DataFrame] = df_negative.randomSplit( Array.fill(nfolds)(1), seed )

    // Generating training and test folds
    val folds: Array[Fold] = (0 until nfolds).map( i =>{
      val trainDf = (dropEle(i,foldsPositive)++dropEle(i,foldsNegative))
        .reduce( (df1,df2) => df1.unionAll(df2) )
      val testDf  = foldsPositive(i).unionAll(foldsNegative(i))

      Fold( abmDataset, numIter, i, trainDf, testDf )
    }).toArray

    abmDataset.folds = abmDataset.folds :+ folds

    // Debug info
    folds.foreach( f => logger.logDebug(f.getSummary()) )

    folds
  }

}