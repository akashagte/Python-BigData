from pyspark.mllib.clustering import KMeans
from numpy import array,random
from math import sqrt
from pyspark import SparkConf,SparkContext
from sklearn.preprocessing import Scale

K=5

#initiate Spark
conf=SparkConf().setMaster("local").setAppName("SparkKMeans")
sc=SparkContext(conf=conf)

#create fake data
def createClusteredData(N,k):
	random.seed(10)
	pointsPerCluster=float(N)/k
	X=[]
	
	for i in range(k):
		incomeCentroid=random.uniform(20000.0,200000.0)
		ageCentroid=random.uniform(20.0,70.0)
		
		for j in range(int(pointsPerCluster)):
			X.append([random.normal(incomeCentroid,10000.0),random.normal(ageCentroid,2.0)])
	X=array(x)
	return X

#load data by normalizing items
data=sc.parallelize(scale(createClusteredData(100,K)))

#Build Model
clusters=KMeans.train(data,K,maxIterations=10,runs=10,initializationMode="random")

#to avoid recalculation use cache() function
resultRDD=data.map(lambda point:clusters.predict(point)).cache()

print("Counts by value:")
counts=resultRDD.countByValue()
print (counts)

print ("\nCluster assignments:")
results=resultRDD.collect()
print (results)

#evaluate model performance by computing Within Set Sum of Squared Errors(WSSSE)
def error(point):
	center=clusters.centers[clusters.predict(point)]
	return sqrt(sum([x**2 for x in (point-center)]))

WSSSE=data.map(lambda point: error(point)).reduce(lambda x,y:x+y)
print("WSSSE for this model="+str(WSSSE))
