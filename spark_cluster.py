from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import json
import numpy as np
import matplotlib.pyplot as plt

conf=SparkConf().setMaster("local").setAppName("cluster")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def jsonFilter(line):
    jsonObject = json.loads(line)
    key = jsonObject.keys()[0]
    values = jsonObject[key]
    if values:
        return True
    else:
        return False

def jsonParseMapper(line):
    jsonObject = json.loads(line)
    key = jsonObject.keys()[0]
    values = jsonObject[key]
    if values:
        return key,values

def valueParse(value):
    jsonObject = json.loads(json.dumps(value))
    price_overview = jsonObject.get('price_overview')
    platformObject = jsonObject.get('platforms')
    genresObject = jsonObject.get('genres')
    categoriesObject = jsonObject.get('categories')
    features = ''
    if price_overview:
        price = json.loads(json.dumps(price_overview)).get('initial',0.1)
    else:
        price = 0.0
    if platformObject:
        platforms = []
        tmp = json.loads(json.dumps(platformObject))
        for key in tmp.keys():
            if tmp[key]:
                features += str(key) + ' '
    if genresObject:
        for value in genresObject:
            genre = str(json.loads(json.dumps(value)).get('description'))
            features += str(genre) + ' '
    if categoriesObject:
        for value in categoriesObject:
            category = str(json.loads(json.dumps(value)).get('description'))
            features += str(category) + ' '
    return price, features

def toCVSLine(data):
    return ','.join(str(d) for d in data)

def data_ETL(raw_data):
    #Filter Data without any content and Map each line to a json object
    data_parse = raw_data.filter(jsonFilter).map(jsonParseMapper)
    #Get interest feature from json object and map them to (appid,price,string_list)
    return data_parse.mapValues(valueParse).map(lambda r : (int(r[0]), float(r[1][0]), r[1][1]))

def create_features(raw_data):
    #Create DataFrame
    data_df = sqlContext.createDataFrame(raw_data.map(lambda r : Row(appid=r[0], price=r[1], sentence=r[2])))
    #Transform sentence into words
    tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
    words_df = tokenizer.transform(data_df)
    #Calculate term frequency
    hashingTF = HashingTF(inputCol='words', outputCol='rawFeatures', numFeatures=5)
    featurized_df = hashingTF.transform(words_df)
    #Calculate inverse document frequency
    idf = IDF(inputCol='rawFeatures', outputCol='features')
    idfModel = idf.fit(featurized_df)
    return idfModel.transform(featurized_df)

def logist(x):
    return 1.0 / (1.0 + np.exp(x))

def features_for_KMeans(row):
    feature = []
    feature.append(row[1])
    for value in row[2]:
        feature.append(value)
    return row[0], np.array(feature)

def spark_KMeans(train_data):
    maxIterations = 10
    runs = 20
    numClusters = [2,3,4,5,6,7,8,9,10,11,12,13,14]
    errors = []
    for k in numClusters:
        model = KMeans.train(train_data, k, maxIterations=maxIterations, runs=runs,initializationMode='random', seed=10, initializationSteps=5, epsilon=1e-4)
        WSSSE = model.computeCost(train_data)
        errors.append(WSSSE)

    plt.plot(numClusters, errors, 'ro')
    plt.xlabel(r'k')
    plt.ylabel(r'inertia')
    plt.title(r'inertia v.s. k')
    plt.savefig('kmeans_cross_validation.png')

    bestModel = KMeans.train(train_data, 6, maxIterations=maxIterations, runs=runs,initializationMode='random', seed=10, initializationSteps=5, epsilon=1e-4)
    return bestModel

def main():
    #Load Data
    raw_data = sc.textFile('appinfo_1000.txt')
    #Data ETL
    data_transformed = data_ETL(raw_data)
    #Extract quantitive feature from strings
    rescaled_df = create_features(data_transformed)
    train_data = rescaled_df.rdd.map(lambda r : (r.appid, r.price, r.features.toArray()))
    #Normalize price to make them have mean 0 and standard deviation 1
    price = train_data.map(lambda r : r[1])
    avg = price.mean()
    sigma = price.stdev()
    X_train = train_data.map(lambda r: (r[0], logist((r[1]-avg)/sigma), r[2])).map(features_for_KMeans).cache()
    X = X_train.map(lambda r : r[1])
    #Use K-Means to cluster data
    bestModel = spark_KMeans(X)
    #Make prediction
    X_pred = X_train.map(lambda r : (r[0], bestModel.predict(r[1])))
    #Save prediction
    X_pred.map(toCVSLine).saveAsTextFile('clustering')

if __name__ == '__main__':
    main()
