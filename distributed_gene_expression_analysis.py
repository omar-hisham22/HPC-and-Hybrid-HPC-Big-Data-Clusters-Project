from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
import pandas as pd

spark = SparkSession.builder.appName("BioinformaticsML").getOrCreate()

df = spark.read.csv("leukemia_expression.csv", header=True, inferSchema=True)
feature_cols = df.columns[:-1]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df)
train_data, test_data = data.randomSplit([0.8, 0.2])

rf = RandomForestClassifier(labelCol=df.columns[-1], featuresCol="features", numTrees=100)
model = rf.fit(train_data)

predictions = model.transform(test_data)
accuracy = predictions.filter(predictions[df.columns[-1]] == predictions.prediction).count() / test_data.count()

print("Accuracy on test data:", accuracy)

spark.stop()