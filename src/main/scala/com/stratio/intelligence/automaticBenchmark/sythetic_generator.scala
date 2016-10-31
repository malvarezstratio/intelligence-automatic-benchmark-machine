package com.stratio.intelligence.automaticBenchmark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
  * Created by malvarez on 27/10/16.
  */
object sythetic_generator extends App {
  val conf = new SparkConf().setAppName("Synthetic generator").setMaster("local[*]")
  val sc = new SparkContext(conf)
  sc.setLogLevel("ERROR")
  val sqlContext = new SQLContext(sc)

  def getBinaryGaussVariable(
                              nH0: Int, nH1: Int, meanH0: Array[Double], meanH1: Array[Double], stdH0: Array[Double], stdH1: Array[Double]
                            ): DataFrame = {

    val distH0: Array[(Double, Double)] = (stdH0, meanH0).zipped.toArray
    val distH1: Array[(Double, Double)] = (stdH1, meanH1).zipped.toArray

    val r = Random
    val saData: IndexedSeq[Row] =
      r.shuffle(
        (for (i <- 1 to nH0) yield
          Row.fromSeq(
            Row(0.0).toSeq ++
              distH0.map {
                case (std, mean) => r.nextGaussian() * std + mean
              }.toSeq)
          ) ++
          (for (i <- 1 to nH1) yield
            Row.fromSeq(
              Row(1.0).toSeq ++
                distH1.map {
                  case (std, mean) => r.nextGaussian() * std + mean
                }.toSeq)
            )
      )

    val data: RDD[Row] = sc.parallelize(saData)

    val newSchema: StructType =
      StructType(
        Array(StructField("label", DoubleType, true)) ++
          (for (i <- 1 to meanH0.length) yield
            StructField("data" + i, DoubleType, true))
      )


    sqlContext.createDataFrame(data, newSchema)
  }

  var nH0 = 100000;  var meanH0 = Array(0.25, 0.2);   var stdH0 = Array(0.05d,0.05d);
  var nH1 = 1000;    var meanH1 = Array(0.25, 0.4);   var stdH1 = Array(0.05d,0.05d);
  val first_quadrant: DataFrame = getBinaryGaussVariable( nH0, nH1, meanH0, meanH1, stdH0, stdH1  )

  nH0 = 1000;  meanH0 = Array(0.75, 0.15);   stdH0 = Array(0.05d,0.05d);
  nH1 = 100;    meanH1 = Array(0.6, 0.2);   stdH1 = Array(0.05d,0.05d);
  val second_quadrant: DataFrame = getBinaryGaussVariable( nH0, nH1, meanH0, meanH1, stdH0, stdH1  )

  nH0 = 1000;  meanH0 = Array(0.75, 0.9);   stdH0 = Array(0.05d,0.05d);
  nH1 = 100000;    meanH1 = Array(0.75, 0.6);   stdH1 = Array(0.05d,0.05d);
  val third_quadrant: DataFrame = getBinaryGaussVariable( nH0, nH1, meanH0, meanH1, stdH0, stdH1  )

  nH0 = 100;  meanH0 = Array(0.75, 0.9);   stdH0 = Array(0.05d,0.05d);
  nH1 = 1000;    meanH1 = Array(0.75, 0.6);   stdH1 = Array(0.05d,0.05d);
  val fourth_quadrant: DataFrame = getBinaryGaussVariable( nH0, nH1, meanH0, meanH1, stdH0, stdH1  )

  /*first_quadrant.show()
  second_quadrant.show()
  third_quadrant.show()
  fourth_quadrant.show()*/

  val fullDataFrame = first_quadrant.unionAll(second_quadrant).unionAll(third_quadrant).unionAll(fourth_quadrant)
  fullDataFrame.rdd.saveAsTextFile("output")
}
