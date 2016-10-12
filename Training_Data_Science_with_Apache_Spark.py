from collections import namedtuple
Person = namedtuple("employee", ["name", "age", "address"])
Address = namedtuple("address", ["city", "country"])

row1 = Person("Bob", 35, Address("London", "UK"))
row2 = Person("Susan", 42, Address("Amsterdam","NL"))
row3 = Person("Sara", 29, Address("Boulder", "USA"))

people = sqlContext.createDataFrame([row1, row2, row3])
display(people)

# Select the people older than 30
people.where(people["age"] > 30)

# Select the name of people from the USA
people.where(people["address.country"]=="USA").select("name")

# Calculate the average age 
import pyspark.sql.functions as func
people.agg(func.avg("age"))


dfsmall = sqlContext.read.parquet("/example.parquet").cache()
print dfsmall.count()

print "dfsmall: {0}".format(dfsmall)
print "\ntype(dfsmall): {0}".format(type(dfsmall))

print dfsmall.scheme, '\n'
dfsmall.printSchema()

from pyspark.sql.types import StructField
help(StructField)

from pyspark.sql.types import StructType, StructField, BooleanType, StringType, LongType
from pyspark.sql import Row

schema = StructType([StructField("title", StringType(), nullable=False, metadata={"language":"English"}),
	                StructField("numberOfEdits", LongType()),
	                StructFiled("redacted", BooleanType())]
)

exampleData = sc.parallelize([Row("Baade's Window", 100, False),
							  Row("Zenomis", 10, True),
							  Row("United States Bureau of Mines", 5280, True)]
)
exampleDF = sqlContext.createDataFrame(exampleData, schema)
display(exampleDF)

exampleDF.printSchema() 
print exampleDF.schema

print exampleDF.schema.fields[0].metadata
print exampleDF.schema.fields[1].metadata

print dfsmall.first()

## Column names
print dfsmall.columns

print dfsmall.drop("text").first()
print dfsmall.select("text").first()[0]

from pyspark.sql.functions import col
errors = dfsmall.fitler(col("title") == '<PARSE ERROR>')
errorCount = errors.count()
print errorCount / float(dfsmall.count())

# we can do the column selection several different ways
print dfsmall.fitler(dfsmall["title"] == '<PARSE ERROR>').count()
print dfsmall.fitler(dfsmall.title == '<PARSE ERROR>').count()

# rename columns 
errors.select(col("title").alias("badTitle")).show(3)
print errors.select("text").first()[0]


(dfsmall
	.select(col("redirect_title").isNotNull().alias("hasRedirect"))
	.groupBy("hasRedirect")
	.count()
	.show()
)

filtered = dfsmall.filter((col("title") != "<PARSE ERROR>") &
						  (col("redirect_title").isNull())&
						  (col("text").isNotNull())
)
print filtered.count()

import pyspark.sql.functions as func
filtered.select("timestamp").show(5)

(filtered
	.select("timestamp", func.data_format("timestamp", "MM/dd/yyyy").alias("data"))
	.show(5)
)

withDate = filtered.withColumn("date", func.date_format("timestamp","MM/dd/yyyy"))
withDate.printSchema()
withDate.select("title", "timestamp", "date").show(3)

# Convert the text field to lowercase
lowered = withDate.select("*", func.lower(col("text")).alias("lowerText")) # Select every column and add a new column, same as withColumn()
print lowered.select("lowerText").first()

print lowered.columns

parsed = (lowered
			.drop("text")
			.drop("timestamp")
			.drop("date")
			.withColumnRenamed("lowerText", "text"))


# RegexTokenizer splits up strings into tokens based on a split pattern. we will split one text on anything that matches one or more non-word character.

from pyspark.ml.feature import RegexTokenizer

tokenizer = (RegexTokenizer()
				.setInputCol("text")
				.setOutputCol("words")
				.setPattern("\\W+")
)
wordsDF = tokenizer.transform(parsed)
wordsDF.select("words").first()

stopwords = set(sc.textFile("/stop_words.txt").collect())
print [word for i, word in zip(range(5), stopwords)]

# Create a function to remove stop words
import re
stopWordsBroadcast = sc.broadcast(stopWords)

def keepWord(word):
	if len(word) < 3:
		return False 

	if word in stopWordsBroadcast.value:
		return False

	if re.search(re.compile(r'[0-9_]'), word):
		return False

	return True

def removeWords(words):
	return [word for word in words if keepWord(word)]

# Create a UDF from our function
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

removeWordsUDF = udf(removeWords, ArrayType(StringType()))

# Scala register returns a UDF
sqlContext.udf.register("removeWords", removeWords, ArrayType(StringType())) 

# Apply our function to the wordsDF DataFrame
noStopWords = (wordsDF
				.withColumn("noStopWords", removeWordsUDF(col("words")))
				.drop("words")
				.withColumnRenamed("noStopWords", "words")
)
noStopWords.cache()

# Calculate the number of words in noStopWords
sized = noStopWords.withColumn("size", func.size("words"))
numberOfWords = sized.agg(func.sum("size").alias("numberOfWords"))
wordCount = numberOfWords.first()[0]

# Compute the word count using select() the function func.explode(), then taking count() on the DataFrame
wordList = noStopWords.select(func.explode("words").alias("word"))
wordListCount = wordList.count()

# Groupby word and count the number of times each word occurs
wordGroupCount = (wordList
					.groupBy("word") # group
					.agg(func.count("word").alias("counts")) # aggregate
					.sort(func.desc("counts")) # sort
)

# we could also use SQL  to accomplish this counting
wordList.registerTempTable("wordList")
wordGroupCount2 = sqlContext.sql("SELECT word, COUNT(word) AS counts FROM wordList GROUP BY word ORDER BY counts DESC")
wordGroupCount2.take(5)

# How many distinct words
distinctWords = wordList.distinct()
distinctWords.count()

#############################################################################################################################
## MLLIB

from pysaprk.mllib import linalg
denseVector = Vectors.dense([1,2,3])
denseVector.dot(denseVector)
Vectors.norm(denseVector,2)

# Convert both sparse and dense vectors to Array calling toArray()
denseArray = denseVector.toArray()

sparseVector = Vectors.sparse(10, [2,7],[1.0, 5.0])

## Inspect is a handy tool for seeing the python source code
import inpsect
print inspect.getsource(SparseVector)

labeledPoint = LabeledPoint(1992, [3.0, 5.5, 10.0])
labeledPoint.features
labeledPoint.label

labeledPointSparse = LabeledPoint(1992, Vectors.sparse(10,{0:3.0, 1:5.5, 2:10.0}))
labeledPointSparse.features
labeledPointSparse.label

## Rating
from pyspark.mllib.recommandation import Rating
rating = Rating(4, 10, 2.0)
rating.user # same as rating[0]
rating.product # same as rating[1]
rating.rating # same as rating[2]

from collections import namedtuple

Address = namedtuple("Address", ["city", "state"])
address = Address("Boulder", "CO")
address.city  # same as address[0]
address.state # same as address[1]

LabelAndFeatures = namedtuple("LabelAndFeatures",["label", "features"])
row1 = LabelAndFeatures(10, Vectors.dense([1.0, 2.0]))
row2 = LabelAndFeatures(20, Vectors.dense([1.5, 2.2]))
df = sqlContext.createDataFrame([row1, row2])

# Create a udf that pulls the first element out of a column that contains DenseVectors
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
firstElement = udf(lambda v: float(v[0]), DoubleType())
df2 = df.select(firstElement("features").alias("first"))
df2.show()

# convert RDD of LabeledPoints to a DataFrame
irisDF = irisRDD.toDF()

class Person(object):
	def __init__(self, name, age):
		self.name = name
		self.age = age

personDF = sqlContext.createDataFrame([Person("Bob", 28), Person("Julie", 35)])

from pyspark.sql.functions import col
irisDFZeroIndex = irisDF.select("features", (col("label")-1).alias("label"))

from pyspark.sql.function import udf
from pysaprk.mllib.linalg import Vectors, VetorUDT
# Take the first two values from a SparseVector and convert them to a DenseVector
firstTwoFeatures = udf(lambda sv: Vectors.dense(sv.toArray()[:2]), VectorUDT())
irisTwoFeatures = irisDFZeroIndex.select(firstTwoFeatures("features").alias("features"), "label").cache()


from pyspark.ml.clustering import KMeans

# Create a KMeans Estimator and set k=3, seed=5, maxIter=20, initSteps=1
kmeans = (KMeans()
			.setK(3)
			.setSeed(5)
			.setMaxIter(20)
			.setInitSteps(1)
) 

# Call fit() on the estimator and pass in our DataFrame
model = kmeans.fit(irisTwoFeatures)

# Obtain the clusterCenters from KMeanModel
centers = model.clusterCenters()

# Use the model to transform the DataFrame by adding cluster predictions
transformed = model.transform(irisTwoFeatures)

print centers

modelCenters = []
iterations = [0,2,4,7,10,20]
for i in iterations:
	kmeans = KMeans(k=3, seed=5, maxIter=i, initStep=1)
	model = kmeans.fit(irisTwoFeatures)
	modelCenters.append(model.clusterCenters())

## Using MLlib instead of ML
# First, convert our DataFrame into an RDD

irisTwoFeaturesRDD = (irisTwoFeatures
						.rdd
						.map(lambda r: (r[1], r[0]))
)
irisTwoFeaturesRDD.take(2)

# Then import MLlib's KMeans as MLlibKMeans to differentiate it from ml.KMeans
from pyspark.mllib.clustering import KMeans as MLlibKMeans
mllibKMeans = MLlibKMeans.train(irisTwoFeaturesRDD.values(), maxIteration=20, seed=5, initializationSteps=1)
print "mllib: {0}".format(mllibKMeans.clusterCenters)
print "ml: {0}".format(centers)

predictionsRDD = mllibKMeans.predict(irisTwoFeaturesRDD.values())
combinedRDD = irisTwoFeaturesRDD.zip(predictionsRDD)
combinedRDD.take(5)

# How do the ml and mllib implementations differ?
import inspect
print inspect.getsource(kmeans.fit)

#################################################################################################################
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import DoubleType

getElement = udf(lambda v. i: float(v[i]), DoubleType())

irisSeparateFeatures = (irisTwoFeatures
							.withColumn("sepalLength", getElement("features", lit(0)))
							.withColumn("sepalWidth", getElement("features", lit(1)))
)

# What about using Column's getItem() method
from pyspark.sql.functions import col
irisTwoFeatures.withColumn("sepalLength", col("features").getItem(0))

# Unfortunately, it doesn't work for vectors, but it does work on arrays
from pyspark.sql import Row
arrayDF = sqlContext.createDataFrame([Row(anArray=[1,2,3]), Row(anArray=[4,5,6])])
arrayDF.show()
arrayDF.select(col("anArray").getItem(0)).show()
arrayDF.select(col("anArray")[1]).show()

# let's register our function and then call it directly from SQL
sqlContext.udf.register("getElement", getElement.func, getElement.returnType)
irisTwoFeatures.registerTempTable("irisTwo")
sqlContext.sql("SELECT getElement(features, 0) AS sepalLength FROM irisTwo")

# Feature Engineering
irisSeparateFeatures.describe("label","sepalLength", "sepalWidth")

from pyspark.ml.feature import StandardScaler
standardScaler = (StandardScaler()
					.setInputCol("features")
					.setOutputCol("standardized")
					.setWithMean(True)
)
print standardScaler.explainParams()

irisStandardizedLength = (standardScaler
							.fit(irisSeparateFeatures)
							.transform(irisSeparateFeatures)
							.withColumn("standardizedLength", getElement("standardized", lit(0)))
)

from pyspark.ml.feature import Normalizer
normalizer = (Normalizer()
				.setInputCol("features")
				.setOutputCol("featureNorm")
				.setP(2.0)
)
irisNormalized = normalizer.transform(irisTwoFeatures)

# Let's just check and see that our norms are equal to 1.0
l2Norm = udf(lambda v: float(v.norm(2.0)), DoubleType)
featureLength = irisNormalized.select(l2Norm("features").alias("featuresLength"),
									  l2Norm("featureNorm").alias("featureNormLength")
)

from pyspark.ml.feature import Bucketizer
splits = [-float('inf'), -0.5, 0.0, 0.5, float('inf')]
lengthBucketizer = (Bucketizer()
						.setInputCol("sepalLength")
						.setOutputCol("lengthFeatures")
						.setSplits(splits)
)
irisBucketizedLength = lengthBucketizer.transform(irisSeparateFeatures)

widthBucketizer = (Bucketizer()
						.setInputCol("sepalWidth")
						.setOutputCol("lengthFeatures")
						.setSplits(splits)
)
irisBucketizedWidth = widthBucketizer.transform(irisBucketizedLength)

from pyspark.ml.pipeline import Pipeline
pipelineBucketizer = Pipeline().setStages([lengthBucketizer, widthBucketizer])
pipelineModelBucketizer = pipelineBucketizer.fit(irisSeparateFeatures)
irisBucketized = pipelineModelBucketizer.transform(irisSeparateFeatures)

from pyspark.ml.feature import VectorAssembler
pipeline = Pipeline()
assembler = VectorAssembler()

print assembler.explainParams()
print "\n ", pipeline.explainParams()

(assembler
	.setInputCols(["lengthFeatures", "widthFeatures"])
	.setOutputCol("featuresBucketized"))

pipeline.setStages([lengthBucketizer, widthBucketizer, assembler])
irisAssembled = pipeline.fit(irisSeparateFeatures).transform(irisSeparateFeatures)

irisSeparateFeatures.groupBy("label").count().orderBy("label")

# Let's build a model that tries to differentiate the first two classes
from pyspark.sql.functions import col
irisTwoClass = irisSeparateFeatures.filter(col("label")<2)
irisTwoClass.groupBy("label").count().orderBy("label")

irisTest, irisTrain = irisTwoClass.randomSplit([0.25, 0.75], seed=0)
irisTest.cache()
irisTrain.cache()

from pyspark.ml.classification import LogisticRegression
lr = (LogisticRegression()
		.setFeaturesCol("featuresBucketized")
		.setRegParam(0.0)
		.setLabelCol("label")
		.setMaxIter(1000))
pipeline.setStages([lengthBucketizer, widthBucketizer, assembler, lr])
pipelineModelLR = pipeline.fit(irisTrain)
irisTestPrediction = (pipelineModelLR
						.transform(irisTest)
						.cache())

print pipelineModelLR.stages
print "\n{0}".format(pipelineModelLR.stages[-1].weights)

from pyspark.ml.feature import OneHotEncoder
oneHotLength = (OneHotEncoder()
					.setInputCol("lengthFeatures")
					.setOutputCol("lengthOneHot"))

pipeline.setStages([lengthBucketizer, widthBucketizer, oneHotLength])
irisWithOneHotLength = pipeline.fit(irisTrain).transform(irisTrain)

oneHotWidth = (OneHotEncoder()
					.setInputCol("widthFeatures")
					.setOutputCol("widthOneHot"))

assembleOneHot = (VectorAssembler()
					.setInputCols(["legnthOneHot", "widthOneHot"])
					.setOutputCol("featuresBucketized"))

pipeline.setStages([lengthBucketizer, widthBucketizer, oneHotLength, oneHotWidth, assembleOneHot])
pipeline.fit(irisTrain).transform(irisTrain)

pipeline.setStages([lengthBucketizer, widthBucketizer, oneHotLength, oneHotWidth, assembleOneHot, lr])
pipelineModelLR2 = pipeline.fit(irisTrain)
irisTestPrediction2 = (pipelineModelLR2
							.transform(irisTest)
							.cache())

logisticModel = pipelineModelLR2.stages[-1]
print logisticModel.intercept
print repr(logisticModel.weights)

# What about model accuracy
from pyspark.sql.functions import col
def modelAccuracy(df):
	return (df.
			.select((col("prediction")==col("label")).cast("int").alias("correct")
			.groupBy()
			.avg("correct")
			.first()[0])

modelOneAccuracy = modelAccuracy(irisTestPrediction)
modelTwoAccuracy = modelAccuracy(irisTestPrediction2)

# Or we can use SQL instead
irisTestPrediction.registerTempTable("modelOnePredictions")
sqlContext.sql("SELECT AVG(int(prediction==label)) FROM modelOnePredictions")

# AUC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
binaryEvaluator = (BinaryClassificationEvaluator()
					.setRawPredictionCol("rawPrediction")
					.setMetricName("areaUnderROC"))

firstModelTestAUC = binaryEvaluator.evaluate(irisTestPrediction)
secondModelTestAUC = binaryEvaluator.evaluate(irisTestPrediction2)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
multiclassEval = MulticlassClassificationEvaluator()
multicalssEval.evaluate(irisTestPrediction)
multicalssEval.evaluate(irisTestPrediction2)

import inspect
print inspect.getsource(MulticlassClassificationEvaluator)

# Using MLlib instead of ML
irisTestPrediction.columns

# Pull the data that we need from our DataFrame and create BinaryClassificationMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics

modelOnePredictionLabel = (irisTestPrediction
							.select("rawPrediction", "label")
							.rdd
							.map(lambda r: (float(r[0][1]), r[1])))

modelTwoPredictionLabel = (irisTestPrediction2
							.select("rawPrediction", "label")
							.rdd
							.map(lambda r: (float(r[0][1]), r[1])))

metricOne = BinaryClassificationMetrics(modelOnePredictionLabel)
metricTwo = BinaryClassificationMetrics(modelTwoPredictionLabel)
print metricOne.areaUnderROC
print metricTwo.areaUnderROC

from pyspark.mllib.regression import LabeledPoint

irisTrainRDD = (irisTrainPredictions
					.select("label", "featuresBucketized")
					.map(lambda r: LabeledPoint(r[0], r[1]))
					.cache())

irisTestRDD = (irisTestPrediction
					.select("label", "featuresBucketized")
					.map(lambda r: (r[0], r[1]))
					.cache())

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
mllibModel = LogisticRegressionWithLBFGS.train(irisTrainRDD, iterations=1000, regParam=0.0)

# Let's calculate the accuracy using RDDs
rddPredictions = mllibModel.predict(irisTestRDD.values())
predictAndLabels = rddPredictions.zip(irisTestRDD.keys())
mllibAccuracy = predictAndLabels.map(lambda (p, l): p==l).mean()

#####################################################################################################################################
## Decision Trees
irisFourFeatures = sqlContext.read.parquet("/example.parquet")

# Convert the data from SparseVector to DenseVector types
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import Vectors, VectorUDT, DenseVector

sparseToDense = udf(lambda sv: Vectors.dense(sv.toArray()), VectorUDT())
irisDense = irisFourFeatures.select(sparseToDense("features").alias("features"), "label")
print "\n".join(map(repr, irisDense.take(2)))

irisTest, irisTrain = irisDense.randomSplit([0.30, 0.70], seed=1)
irisTest.cache()
irisTrain.cache()

# Update the Metadata for Decision Trees and build a tree
# we use StringIndexer on our labels in order to obtain a DataFrame that decision trees can work with
from pyspark.ml.feature import StringIndexer
stringIndexer = (StringIndexer()
					.setInputCol("label")
					.setOutputCol("indexed"))

indexerModel = stringIndexer.fit(irisTrain)
irisTrainIndexed = indexerModel.transform(irisTrain)

print irisTrainIndexed.schema.fields[1].metadata
print irisTrainIndexed.schema.fields[2].metadata

# Let's build a decision tree to classify our data
from pyspark.ml.classification import DecisionTreeClassifier

dt = (DecisionTreeClassifier()
		.setLabelCol("indexed")
		.setMaxDepth(5)
		.setMaxBins(10)
		.setImpurity("gini"))

print dt.explainParams("impurity")
print "\n", dt.explainParams("maxBins")

# Fit the model and display predictions on the test data
dtModel = df.fit(irisTrainIndexed)
predictionsTest = dtModel.transform(indexerModel.transform(irisTest))

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
multiEval = (MulticlassClassificationEvaluator()
				.setMetricName("precision")
				.setLabelCol("indexed"))

# View the decision tree model
dtModelString = dtModel.toDebugString()

readableModel = dtModelString
for feature, name in enumerate(["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]):
	readableModel = readableModel.replace("feature {0}".format(feature), name)
print readableModel

# Cross-Validation
 from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
 from pyspark.ml.pipeline import Pipeline

 cvPipeline = Pipeline().setStages([stringIndexer, dt])

 multiEval.setMetricName("precision")

 paramGrid = (ParamGridBuilder()
 				.addGrid(dt.maxDepth, [2,4,6,10])
 				.build())

cv = (CrossValidator
		.setEstimator(cvPipeline)
		.setEvaluator(multiEval)
		.setEstimatorParamMaps(paramGrid)
		.setNumFolds(5))

cvModel = cv.fit(irisTrain)
predictions = cvModel.transform(irisTest)
print multiEval.evaluate(predictions)

# What was our best model
bestDTModel = cvModel.bestModel.stages[-1]

print bestDTModel._java_obj.parent().explainParams()
print bestDTModel._java_obj.parent().getMaxDepth()

# Random Forest and PolynomialExpansion
px = (PolynomialExpansion()
		.setInputCol("features")
		.setOutputCol("polyFeatures"))
print px.explainParams()

from pyspark.ml.classification import RandomForestClassifier
rf = (RandomForestClassifier()
		.setLabelCol("indexed")
		.setFeaturesCol("polyfeatures"))
print rf.explainParams()

(rf
	.setMaxBins(10)
	.setMaxDepth(2)
	.setNumTrees(20)
	.setSeed(0))

rfPipeline = Pipeline().setStages([stringIndexer, px, rf])
rfModelPipeline = rfPipeline.fit(irisTrain)
rfPredictions = rfModelPipeline.transform(irisTest)
print multiEval.evaluate(rfPrediction)

paramGridRand = (ParamGridBuilder()
					.addGrid(rf.maxDepth, [2,4,8,12])
					.baseOn(rf.numTrees, 20)
					.build())

cvRand = (CrossValidator()
			.setEstimator(rfPipeline)
			.setEvaluator(multiEval)
			.setEstimatorParamMaps(paramGridRand)
			.setNumFolds(2))

cvModelRand = cvRand.fit(irisTrain)
predictionsRand = cvModelRand.transform(irisTest)
print multiEval.evaluate(predictionsRand)
print cvModelRand.bestModel.stages[-1]._java_obj.parent().getMaxDepth()
print cvModelRand.bestModel.stages[-1]._java_obj.toDebugString()

####################################################################################################################
## Scikit-Learn
import numpy as np
import sklearn import datasets

# load the data
iris = datasets.load_iris()

# Generate test and train sets
size = len(iris.target)
indices = np.random.permutation(size)

cutoff = int(size * 0.3)

testX = iris.data[indices[0:cuttoff], :]
trainX = iris.data[indices[cutoff:], :]
testY = iris.target[indices[0:cutoff]]
trainY = iris.target[indices[cutoff:]]

from sklearn.neighbors import KNeighborsClassifier
# Create a KNeighborsClassifier using the default settings
knn = KNeighborsClassifier()
knn.fit(trainX, trainY)

predictions = knn.predict(testX)
# print out the accuracy of the classifier on the test set
print sum(predictions==testY) / float(len(testY))

# Grid Search
from sklearn.cross_validation import train_test_split

def runNearestNeighbors(k):
	irisData = dataset.load_iris()
	yTrain, yTest, XTrain, XTest = train_test_split(irisData.target, irisData.data)
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(XTrain, yTrain)
	predictions = knn.predict(XTest)
	accuracy = (predictions == yTest).sum() / float(len(yTest))
	return (k, accuracy)

k = sc.parallelize(range(1,11))
results = k.map(runNearestNeighbors)
print "\n".join(map(str, results.collect()))

# Let's transfer the data using a Broadcast instead of loading it at each executor.
irisBroadcast = sc.broadcast(iris)

def runNearestNeighborsBroadcast(k):
	irisData = dataset.load_iris()
	yTrain, yTest, XTrain, XTest = train_test_split(irisBroadcast.value.target, irisBroadcast.value.data)
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(XTrain, yTrain)
	predictions = knn.predict(XTest)
	accuracy = (predictions == yTest).sum() / float(len(yTest))
	return (k, accuracy)

from sklearn.cross_validation import KFold 
# Create indices for 10-fold cross validation
kf = KFold(size, n_folds=10)
folds = sc.parallelize(kf)
print folds.take(1)

import numpy as np

def runNearestNeighborsWithFolds((trainIndex, testIndex)):
	XTrain = irisBroadcast,value.data[trainIndex]
	yTrain = irisBroadcast,value.target[trainIndex]
	XTest = irisBroadcast,value.data[testIndex]
	yTest = irisBroadcast,value.target[testIndex]

	knn =KNeighborsClassifier(n_neighbors=5)
	knn.fit(XTrain, yTrain)
	predictions = knn.predict(XTest)
	correct = (predictions==yTest).sum()
	total = len(testIndex)
	return np.array([correct, total])

# Run nearest neighbors on each of the folds
foldResults = folds.map(runNearestNeighborsWithFolds)
print "correct / total" + "\n".join(map(str, foldResults.collect()))

# Note that using sum() on a RDD of numpy arrays sums by columns
correct, total = foldResults.sum()
print correct / float(total)

## Sampling
# Split the iris dataset into 8 partitions
irisData = sc.parallelize(zip(iris.target, iris.data), 8)
print irisData.take(2)

# View the number of elements found in each of the eight partitions
irisData
	.mapPartitions(lambda x: [(len(list(x)))])
	.collect()
# View the target (y) stored by partition
print "\n", irisData.keys().glom().collect()

# Using partitionBy so that the data is randomly ordered across partitions
randomOrderData = (irisData
					.map(lambda x: (np.random.randint(5), x))
					.partitionBy(5)
					.values())

# Show the new groupings of target variables
print randomOrderData.keys().glom().collect()

print randomOrderData.keys().mapPartitions(lambda x: [len(list(x))]).collect()

def runNearestNeighborsPartition(LabelAndFeatures):
	y, X = zip(*LabelAndFeatures)
	yTrain, yTest, XTrain, XTest = train_test_split(y,X)
	knn = KNeighborsClassifier()
	knn.fit(XTrain, yTrain)
	predictions = knn.predict(XTest)
	correct = (predictions == yTest).sum()
	total = len(yTest)
	return (np.array([correct, total]))

sampleResults = randomOrderData.mapPartitions(runNearestNeighborsPartition)














