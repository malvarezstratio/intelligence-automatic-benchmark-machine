package com.stratio.intelligence.automaticBenchmark

import org.apache.spark.SparkConf


class ABMTest extends ABMFunSuite{

  override val conf =
    new SparkConf().setAppName("Automatic Benchmark Machine")
    .setMaster("local")
    .set("spark.executor.memory", "8g")
    .set("spark.driver.memory", "8g")


}
