'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-30 11:00:51
@LastEditors: Please set LastEditors
@Description: 
'''
import time
import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.types as typ
import pyspark.ml.feature as ft
import pyspark.ml.classification as cl
import pyspark.ml.evaluation as ev
import pyspark.ml.tuning as tune
from pyspark.ml import Pipeline

LABELS = [  # 数据字段
    ('INFANT_ALIVE_AT_REPORT', typ.IntegerType()),
    ('BIRTH_PLACE', typ.StringType()),
    ('MOTHER_AGE_YEARS', typ.IntegerType()),
    ('FATHER_COMBINED_AGE', typ.IntegerType()),
    ('CIG_BEFORE', typ.IntegerType()),
    ('CIG_1_TRI', typ.IntegerType()),
    ('CIG_2_TRI', typ.IntegerType()),
    ('CIG_3_TRI', typ.IntegerType()),
    ('MOTHER_HEIGHT_IN', typ.IntegerType()),
    ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
    ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
    ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
    ('DIABETES_PRE', typ.IntegerType()),
    ('DIABETES_GEST', typ.IntegerType()),
    ('HYP_TENS_PRE', typ.IntegerType()),
    ('HYP_TENS_GEST', typ.IntegerType()),
    ('PREV_BIRTH_PRETERM', typ.IntegerType())
]

def create_spark_session():
    conf = SparkConf().setAppName("ML LR").set(
        "spark.ui.shaowConsoleProgress", 'false')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    print("Master: {}".format(sc.master))
    set_logger(sc)
    set_path(sc)
    return spark


def set_logger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger('org').setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger('akka').setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    sc.setLogLevel("FATAL")


def set_path(sc):
    global PATH
    if sc.master[:5] == 'local':
        PATH = "/mnt/data1/workspace/data_analysis_mining/learningPySpark/Chapter06"
    else:
        PATH = "hdfs://master:9000/user/hduser"

def prepare_data(spark):
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    schema = typ.StructType(
        [typ.StructField(e[0], e[1], False) for e in LABELS]
    )
    # 创建DataFrame
    births = spark.read.csv("births_transformed.csv.gz", 
                            header=True, schema=schema)
    # BIRTH_PLACE 转为int 型
    births = births.withColumn("BIRTH_PLACE_INT",
                     births['BIRTH_PLACE'].cast(typ.IntegerType()))
    births_train, births_test = births.randomSplit([0.7, 0.3], seed=666)
    print("将数据分trainData: {}, testData: {}".format(
        births_train.count(), births_test.count()
    ))
    return births_train, births_test

def train_evaluate(train_data, test_data):
    # 再使用onehot编码
    encoder = ft.OneHotEncoder(inputCol="BIRTH_PLACE_INT", 
                               outputCol="BIRTH_PLACE_VEC")
    # 聚合特征向量
    features_creator = ft.VectorAssembler(
        inputCols=[col[0] for col in LABELS[2:]] + [encoder.getOutputCol()],
        outputCol='features'
        )
    data_pipeline = Pipeline(stages=[encoder, features_creator])
    data_transformer = data_pipeline.fit(train_data)
    # 分类器
    logistic = cl.LogisticRegression(labelCol="INFANT_ALIVE_AT_REPORT")
    
    # 评估结果
    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol="probability",
        labelCol="INFANT_ALIVE_AT_REPORT")

    # grid search 超参数调优
    grid = tune.ParamGridBuilder()\
        .addGrid(logistic.maxIter, [5, 10, 20, 50, 100])\
        .addGrid(logistic.regParam, [0.01, 0.05, 0.1, 0.5, 1.])\
        .build()
    
    # 交叉验证
    cv = tune.CrossValidator(
        estimator=logistic,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3)
    
    # 建立pipeline
    pipeline = Pipeline(stages=[data_transformer, cv])
    pipeline_model = pipeline.fit(train_data)
    
    cv_model = pipeline_model.stages[-1]
    best_param = get_best_param(cv_model)


    AUC, AP = evaluate_model(pipeline_model, test_data)
    
    return AUC, AP, best_param, pipeline_model


def get_best_param(cv_pipeline_model):
    result = [
        (
            [
                {key.name, param_value} for key, param_value in zip(
                    param.keys(), param.values())
            ], metric
        ) for param, metric in zip(
            cv_pipeline_model.getEstimatorParamMaps(),
            cv_pipeline_model.avgMetrics)
    ]
    # ([{'maxIter': 50}, {'regParam': 0.01}], 0.7385557487596289)
    best_param = sorted(result, key=lambda e: e[1], reverse=True)[0]
    return best_param[0]


def evaluate_model(pipeline_model, data):
    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol="probability",
        labelCol="INFANT_ALIVE_AT_REPORT")
    results = pipeline_model.transform(data)
    AUC = evaluator.evaluate(results, 
                            {evaluator.metricName: 'areaUnderROC'})
    AP = evaluator.evaluate(results, 
                            {evaluator.metricName: 'areaUnderPR'})
    return AUC, AP


def save_model(model, model_path):
    model.write().overwrite().save(
        os.path.join(PATH, model_path)
    )

if __name__ == "__main__":
    print("Start ML Logisitc Regression")
    spark = create_spark_session()
    print("==========数据准备阶段===============")
    births_train, births_test = prepare_data(spark)
    births_train.persist()
    births_test.persist()
    print("==========训练评估阶段===============")
    AUC, AP, best_param, pipeline_model = train_evaluate(
        births_train, births_test)
    print("最佳模型使用的参数{}, 测试集: AUC={}, AP={}".format(
        best_param, AUC, AP))

    # print("==========测试阶段===============")
    # AUC, AP = evaluate_model(pipeline_model, births_test)
    
    
    births_train.unpersist()
    births_test.unpersist()
    save_model(pipeline_model, 'ML_LR')
    