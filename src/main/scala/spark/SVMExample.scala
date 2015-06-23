package spark

import org.apache.spark._
import org.apache.spark.mllib.classification.{SVMModel,SVMWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

/**
 * @author dpranantha
 */
object SVMExample {
  val dataFile = "data/iris.data" 
  val modelFile = "models/svm.model"
  lazy val sc = new SparkContext(new SparkConf().setAppName("SVMExample").setMaster("local"))

  def main(args: Array[String]) = {
    //read data
    val data = MLUtils.loadLibSVMFile(sc,dataFile).cache().filter { x => x.label != 2 }
    println("Size: "+ data.cache().count())

    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val (trainingData, testData) = (splits(0).cache(), splits(1))

    // Train an SVM model.
    val numIter = 100

    val model = SVMWithSGD.train(trainingData, numIter);

   // model.clearThreshold()

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + (testErr*100) + "%")

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)
//    // Save and load model
//    model.save(sc, modelFile)
//    val sameModel = SVMModel.load(sc, modelFile)
  }
}