/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.gmu.stc

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.{SparkConf, SparkContext}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image.GreyImgNormalizer
import com.intel.analytics.bigdl.models.lenet.{LeNet5, Utils}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.numeric.NumericFloat
import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Validator}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import Utils._

object LeNetTest {
  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.disableCheckSysEnv", "true")

    val conf = Engine.createSparkConf().setMaster("local[4]")
      .setAppName("Train Lenet on MNIST")
      .set("spark.task.maxFailures", "1")
      .set("bigdl.disableCheckSysEnv", "true")
      .set("BIGDL_LOCAL_MODE", "1")
      .set("DL_ENGINE_TYPE", "mklblas")
      .set("spark.executorEnv.DL_ENGINE_TYPE", "mklblas")

    val sc = new SparkContext(conf)
    Engine.init

    val classNum = 3

    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100))
      .add(Tanh())
      .add(Linear(100, classNum))
      .add(LogSoftMax())

    val trainData = "/Users/feihu/Documents/IDEAProjects/BigDL/data/MNIST/training/train-images-idx3-ubyte"
    val trainLabel = "/Users/feihu/Documents/IDEAProjects/BigDL/data/MNIST/training/train-labels-idx1-ubyte"



  }
}
