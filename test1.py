from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import RFormula
from jpmml_sparkml import toPMMLBytes
from pyspark.shell import spark
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Binarizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import PCA

df = spark.read.csv("/home/sameera/Downloads/iris.csv", header = True, inferSchema = True)

assembler = VectorAssembler(
    inputCols=["SepalWidth", "SepalLength", "PetalLength", "PetalWidth"],
    outputCol="features")
output = assembler.transform(df)


slicer = VectorSlicer(inputCol="features", outputCol="features2", indices=[1,2,3])
output1 = slicer.transform(output)


pca = PCA(k=2, inputCol="features2", outputCol="selectedFeatures")
result = pca.fit(output1).transform(output1)


binarizer = Binarizer(threshold=10.3, inputCol="PetalLength", outputCol="binarized_feature")
binarizedDataFrame = binarizer.transform(result)
binarizedDataFrame.show()


assembler2 = VectorAssembler(
    inputCols=["selectedFeatures", "binarized_feature"],
    outputCol="features3")
output2 = assembler2.transform(binarizedDataFrame)
output2.show()


formula = RFormula(
    formula="Species ~ features3",
    featuresCol="features4",
    labelCol="label")
classifier = DecisionTreeClassifier()


pipeline = Pipeline(stages = [assembler, slicer,pca, binarizer,assembler2, formula,classifier])
pipelineModel = pipeline.fit(df)
pmmlBytes = toPMMLBytes(spark, df, pipelineModel)
with open('test1.pmml','wb') as output:
    output.write( pmmlBytes)
print(pmmlBytes)
