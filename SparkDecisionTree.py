#import Libraries
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

#Spark initiation
conf=SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc=SparkContext(conf=conf)


#featureConversion
def binary(YN):
	if(YN=='Y'):
		return 1
	else:
		return 0
	
def mapEducation(degree):
	if(degree=='BS'):
		return 1
	elif(degree=='MS'):
		return 2
	elif(degree=='PhD'):
		return 3
	else:
		return 0

#convert a list of raw fields from csv to a LabeledPoint that MLLib can use
#all data must be numerical
def createLabeledPoints(fields):
	yearsExperience=int(fields[0])
	employed=binary(fields[1])
	previousEmployers=int(fields[2])
	educationLevel=mapEducation(fields[3])
	topTier=binary(fields[4])
	interned=binary(fields[5])
	hired=binary(fields[6])
	
	return LabeledPoint(hired,array([yearsExperience,employed,previousEmployers,educationLevel,topTier,interned]))

#import Data file	
rawData=sc.textFile("C:\Users\Akash Agte\Downloads\Learning\Python\DataScience\PastHires.csv")
header=rawData.first()
rawData=rawData.filter(lambda x:x!=header)

#split each line into a list of comma delimiters
csvData=rawData.map(lambda x:x.split(","))

#convert lists (attributes) into LabeledPoints
trainingData=csvData.map(createLabeledPoints)


#Create a test candidate
testCandidates=[array([10,1,3,1,0,0],[9,0,2,1,0,1])]
testData=sc.parallelize(testCandidates)


#train decisionTreeClassifier using our dataset
model=DecisionTree.trainClassifier(trainingData,numClasses=2,categoricalFeaturesInfo={1:2,3:4,4:2,5:2},impurity='gini',maxDepth=5,maxBins=32)

#Predictions
predictions=model.predict(testData)
print('Hire Prediction:')
results=predictions.collect()

for result in results:
	print result

#print decision tree here
print('Learned classification tree model:')
print(model.toDebugString())

								   