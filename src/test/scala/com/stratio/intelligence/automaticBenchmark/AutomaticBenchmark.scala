package com.stratio.intelligence.automaticBenchmark

import com.stratio.intelligence.automaticBenchmark.models.BenchmarkModel
import com.stratio.intelligence.automaticBenchmark.models.decisionTree.{BenchmarkDecisionTree, DTParams}
import com.stratio.intelligence.automaticBenchmark.models.logisticModelTree.{BenchmarkLMT, LMTParams}
import com.stratio.intelligence.automaticBenchmark.models.logisticRegression.{LRParams, BenchmarkLogisticRegression}
import com.stratio.intelligence.automaticBenchmark.output.OutputConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object AutomaticBenchmark extends App {

  // TODO - Parametrizar la eliminacion o no de una de las categorias en el One Hot
  // TODO - Generalizar a clasificacion multiclase/regression
  // TODO - Incluir el código del LMT en vez del JAR
  // BUg - pima.indians el LMT, dependiendo de cada ejecución, puede funcionar o no. No crece hijos

  // Create sparkContext and sqlContext
  val conf = new SparkConf().setAppName("Automatic Benchmark Machine")
    .setMaster("local")
    .set("spark.executor.memory", "8g")
    .set("spark.driver.memory", "8g")
  val sc = new SparkContext(conf)
  sc.setLogLevel("ERROR")
  val sqlContext = new SQLContext(sc)

  // Data files and description files

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


    // LOCAL data files
      /*
      val datafile1 = "./src/main/resources/diagnosis.csv"
      val descriptionfile1 = "./src/main/resources/diagnosis.description"

      val datafile2 = "./src/main/resources/bank.csv"
      val descriptionfile2 = "./src/main/resources/bank.description"
      */

  // New Automatic Bechmark Machine
    val abm = new AutomaticBenchmarkMachine(sqlContext)
    abm.enableDebugMode()

  // Defining models
    val models:Array[BenchmarkModel] = Array(
      new BenchmarkLogisticRegression()
          .setParameters( LRParams().setFitIntercept(true) ),
      new BenchmarkDecisionTree()
        .setParameters( DTParams().setMaxBins(100) ),
      new BenchmarkLMT()
        .setParameters(
            LMTParams()
              .setMaxBins(100)
              .setDebugConsole(true)
              .setMaxPointsForLocalRegression(100000)
              .setMinElements(-1)
              .setSeed(12)
              .setPruningRatio(0)
        )
    )

  // Executing the benchmarking process
    abm.run(
      dataAndDescriptionFiles =
        Array(
          // (datafile1,descriptionfile1)
           (datafile5,descriptionfile5)
          // (datafile3,descriptionfile3),
          // (datafile4,descriptionfile4),
          // (datafile5,descriptionfile5)
        ),
      outputConf =
        OutputConf().setFilePath("myoutput.txt").setShowTrainedModel(true),
      seed = 11,
      kfolds = 3,
      mtimesFolds = 1,
      algorithms = models
    )

}
