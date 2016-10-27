package com.stratio.intelligence.automaticBenchmark.models

trait MLModel{

  // All these should be defined only once, in the abstract superclass as public final symbolic constants
  val m_KEY_FULLDF_TRAIN            = "fullDFtrain"
  val m_KEY_FULLDF_TEST             = "fullDFtest"
  val m_KEY_RDDCATEG_TRAIN          = "RDDcategTrain"
  val m_KEY_RDDCATEG_TEST           = "RDDcategTest"
  val m_KEY_RDDBINARY_TRAIN         = "RDDbinaryTrain"
  val m_KEY_RDDBINARY_TEST          = "RDDbinaryTest"
  val m_COMPRESSED_CATEG_FEATURES   = "featuresCateg"
  val m_COMPRESSED_BINARY_FEATURES  = "featuresBinary"
  val m_KEY_FEATURES_MAP            = "categoricalFeaturesMap"
  val m_classColumn                 = "classColumn"

  def train() = ???

  def predict() = ???

  def getMetrics() = ???

}
