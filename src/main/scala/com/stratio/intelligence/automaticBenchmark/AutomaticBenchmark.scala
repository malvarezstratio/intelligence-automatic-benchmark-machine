package com.stratio.intelligence.automaticBenchmark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object AutomaticBenchmark extends App {

  // Create sparkContext and sqlContext
  val conf = new SparkConf().setAppName("Automatic Benchmark Machine").setMaster("local[*]")
  val sc = new SparkContext(conf)
  sc.setLogLevel("ERROR")
  val sqlContext = new SQLContext(sc)


  // val data: RDD[String] = sc.textFile("hdfs://144.76.3.23:54310/data/benchmarks/bank/bank.description")
  //  data.take(10).foreach(println)

  // Data files and description files
/*
    // HDFS data files
    val datafile1 = "hdfs://144.76.3.23:54310/data/benchmarks/default_credit_cards/default-credit-cards-no-header.data"
    val descriptionfile1 = "hdfs://144.76.3.23:54310/data/benchmarks/default_credit_cards/default-credit.description"

    val datafile2 = "hdfs://144.76.3.23:54310/data/benchmarks/bank/bank.csv"
    val descriptionfile2 = "hdfs://144.76.3.23:54310/data/benchmarks/bank/bank.description"

    val datafile3 = "hdfs://144.76.3.23:54310/data/benchmarks/adults/adults-labeled.csv"
    val descriptionfile3 = "hdfs://144.76.3.23:54310/data/benchmarks/adults/adults.description"

    val datafile4 = "hdfs://144.76.3.23:54310/data/benchmarks/bikes/bikes_bien2_comas.csv"
    val descriptionfile4 = "hdfs://144.76.3.23:54310/data/benchmarks/bikes/bikes_bien2_comas.description"

    val datafile5 = "hdfs://144.76.3.23:54310/data/benchmarks/pima-indians/pima-indians-full.csv"
    val descriptionfile5 = "hdfs://144.76.3.23:54310/data/benchmarks/pima-indians/pima-indians-full.description"
    */

  // LOCAL data files
  val datafile1 = "./src/main/resources/diagnosis.csv"
  val descriptionfile1 = "./src/main/resources/diagnosis.description"


  // New Automatic Bechmark Machine
    val abm = new AutomaticBenchmarkMachine(sqlContext)

    abm.run(
      dataAndDescriptionFiles = Array((datafile1,descriptionfile1) ),
      outputFile = "myoutput.txt",
      seed = 11,
      kfolds = 3,
      mtimesFolds = 1,
      algorithms = Array("LMT")
    )

}
