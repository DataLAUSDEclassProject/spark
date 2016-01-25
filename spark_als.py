from pyspark.ml import Pipeline
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.recommendation import ALS, Rating
import json
import numpy as np
import matplotlib.pyplot as plt

conf=SparkConf().setMaster("local").setAppName("als")
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
    return jsonObject["playtime_forever"],jsonObject["appid"]

def data_ETL(raw_data):
    #Map each line to json object and extract features and transform them to (user_id, app_id,play_time)
    data_transformed = raw_data.filter(jsonFilter).map(jsonParseMapper).flatMapValues(lambda x : x)
    return data_transformed.mapValues(valueParse).map(lambda x : (x[0], x[1][0], x[1][1]))

def data_reindex(raw_data):
    #Reindex user_id to (0,1,2,...,n)
    #Reindex app_id to (0,1,2,...,m)
    users = set(raw_data.keys().collect())
    users_index = dict((x,y) for x,y in zip(users,range(len(users))))
    indexed_users = raw_data.map(lambda x : (users_index[x[0]], x[1], x[2]))
    rating_new = indexed_users.map(lambda x : (x[2], x[1], x[0]))
    items = set(rating_new.keys().collect())
    item_index = dict((x,y) for x,y in zip(items,range(len(items))))
    return rating_new.map(lambda x :(int(x[2]), int(item_index[x[0]]), float(x[1])))

def toCVSLine(data):
    return ','.join(str(d) for d in data)

def rates_transform(rates_data):
    rates = rates_data.map(lambda x : float(x[2]))
    #mean = rates.mean()
    #sigma = rates.stdev()
    #print mean, sigma
    return rates_data.map(lambda x : (int(x[0]), int(x[1]), x[2]))

def als(data):
    train, test = data.randomSplit(weights=[0.8, 0.2])
    X_train = train.map(lambda r : Rating(r[0], r[1], r[2]))
    y = test.map(lambda r : ((r[0], r[1]), r[2]))
    X_test = test.map(lambda r : (r[0], r[1]))
    rank = 7
    X_train.cache()
    X_test.cache()
    lambdas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    numIterations = 10
    nonnegative=True
    bestModel = None
    error = float('Inf')
    errors = []
    #Use ALS to predict play time for test users and choose the best parameter for lambda
    for lmbda in lambdas:
        model = ALS.train(X_train, rank, numIterations, lmbda, nonnegative=nonnegative)
        y_hat = model.predictAll(X_test).map(lambda r : ((r[0], r[1]), r[2]))
        ratesAndPreds = y.join(y_hat)
        MSE = ratesAndPreds.map(lambda r : ((r[1][0]) - (r[1][1]))**2).mean()
        errors.append(MSE)
        if MSE < error:
            bestModel = model
            error = MSE
    #Plot mean square error v.s. lambda
    plt.plot(lambdas, errors, 'ro')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$MSE$')
    plt.title(r'MSE v.s. $\lambda$')
    plt.savefig('cross_validation_p.png')
    #Make Prediction by using the best model
    y_hat = model.predictAll(X_test).map(lambda r : (r[0], r[1], r[2]))
    y_hat.map(toCVSLine).saveAsTextFile('prediction')
    return bestModel, error

def main():
    #Load Data
    raw_data = sc.textFile('inventory_1000.txt')
    #Data ETL
    data = data_ETL(raw_data)
    #Reindex user_id and app_id
    indexed_data = data_reindex(data)
    transformed_data = rates_transform(indexed_data)
    als(transformed_data)

if __name__ == '__main__':
    main()
