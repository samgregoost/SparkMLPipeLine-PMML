from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import RFormula
from jpmml_sparkml import toPMMLBytes
from pyspark.shell import spark

df0 = spark.read.csv("/home/sameera/Downloads/iris.csv", header = True, inferSchema = True)

formula0 = RFormula(formula = "Species ~ SepalWidth + SepalWidth + PetalLength + PetalWidth")

model0 = formula0.fit(df0)
df = model0.transform(df0)

#formula = RFormula(formula = "Species ~ SepalWidth : SepalLength")
binarizer = Binarizer(threshold=1.0, inputCol="SepalWidth", outputCol="features2")

#model = binarizer.fit(df)
df2 = binarizer.transform(df)
formula = RFormula(formula = "Species ~ SepalWidth + features2")
model = formula.fit(df2)
model.transform(df2).show()
classifier = DecisionTreeClassifier()
pipeline = Pipeline(stages = [formula0, binarizer, formula, classifier])
pipelineModel = pipeline.fit(df0)

pmmlBytes = toPMMLBytes(spark, df0, pipelineModel)
with open('test.pmml', 'wb') as output:
    output.write( pmmlBytes)
print(pmmlBytes)
