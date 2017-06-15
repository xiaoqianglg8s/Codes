from pyspark import SparkConf, SparkContext
import operator

conf = SparkConf().setMaster("local[*]").setAppName("First_App")

sc = SparkContext(conf=conf)