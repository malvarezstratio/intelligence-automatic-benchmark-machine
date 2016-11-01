package com.stratio.intelligence.automaticBenchmark.output


case class OutputConf() {

  // Property: filePath: path for writing output summary text file
    private[this] var _filePath: String = _
    def filePath: String = _filePath
    def setFilePath(value: String): OutputConf = { _filePath = value; this  }

}
