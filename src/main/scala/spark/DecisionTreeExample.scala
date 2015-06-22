package spark

import org.apache.spark._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object DecisionTreeExample {
  val dataFile = "data/iris.data" 
  val modelFile = "models/dt.model"
  lazy val sc = new SparkContext(new SparkConf().setAppName("DecisionTreeExample").setMaster("local"))

  def main(args: Array[String]) = {
    //read data
    val data = MLUtils.loadLibSVMFile(sc,dataFile).cache()

    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 3
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "entropy"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)

//    // Save and load model
//    model.save(sc, modelFile)
//    val sameModel = DecisionTreeModel.load(sc, modelFile)
  }
}