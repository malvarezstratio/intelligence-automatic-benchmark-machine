val a = 1 + 1
val b = a + 3


case class A( a:String, b:String)

"b" match {
  case "a" => println("ok")
  case _ => println("ko")
}