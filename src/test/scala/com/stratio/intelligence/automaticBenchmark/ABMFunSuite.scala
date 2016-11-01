package com.stratio.intelligence.automaticBenchmark

import com.holdenkarau.spark.testing._
import grizzled.slf4j.Logging
import org.apache.spark.sql.SQLContext
import org.scalatest.{FunSuite, Outcome}

/** Base abstract class for all unit tests in GTB analysis project */
trait ABMFunSuite extends FunSuite
  with SharedSparkContext
  with Logging {

  def sqlContext: SQLContext = ABMFunSuiteContextProvider._sqlContext

  override def beforeAll() {
    super.beforeAll()
    ABMFunSuiteContextProvider._sqlContext = new SQLContext(sc)
  }

  override def afterAll() {
    super.afterAll()
    ABMFunSuiteContextProvider._sqlContext = null
  }

  /**
    * Log the suite name and the test name before and after each test.
    *
    * Subclasses should never override this method. If they wish to run
    * custom code before and after each test, they should mix in the
    * {{org.scalatest.BeforeAndAfter}} trait instead.
    */
  final protected override def withFixture(test: NoArgTest): Outcome = {
    val testName = test.text
    val suiteName = this.getClass.getName
    val shortSuiteName = suiteName.replaceAll("com.stratio.gtbAnalysis", "c.s.g")
    try {
      logger.info(s"\n\n===== TEST OUTPUT FOR $shortSuiteName: '$testName' =====\n")
      test()

    } finally {
      logger.info(s"\n\n===== FINISHED $shortSuiteName: '$testName' =====\n")
    }
  }

}

object ABMFunSuiteContextProvider {
  @transient var _sqlContext: SQLContext = _
}