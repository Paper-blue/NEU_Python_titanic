import pandas
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pandas.read_csv("D:/python代码/train.csv")

print(titanic.head(5))
print(titanic.describe())

# 数据集的Age列只包含714行，而所有其他列都有891行。
# Age属性十分关键，用原来数据进行填充
#考虑使用机器学习的办法，但训练样本 太少，选用均值填充

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

#查看年龄与存活的关系
plt.figsize = (18,4)
titanic['Age_int'] = titanic['Age'].astype(int)
average_age = titanic[['Age_int','Survived']].groupby(['Age_int'],as_index = False).mean()
sns.barplot(x= 'Age_int',y ='Survived',data = average_age)



#查看性别与存活的关系
titanic.groupby(['Sex','Survived'])['Survived'].count()
titanic[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

#将性别转换为数值，男为0，女为1

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1


# 从sklearn导入用于交叉验证的线性回归类和助手
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

#用于预测的列
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#初始化算法类
alg = LinearRegression()

#设置random_state以确保每次运行时都得到相同的分割。
kf = KFold(titanic.shape[0], n_splits=3, random_state=1)

predictions = []
#为数据集生成交叉验证折叠。它返回与列车和测试折叠相对应的行索引。
#每次在一个折叠上训练模型，在另一个折叠上测试它。总共有3个折叠，所以有3个组合
#一次训练和测试两次。
for train, test in kf.split(titanic):
    #预测因素
    train_predictors = (titanic[predictors].iloc[train, :])
    #预测目标
    train_target = titanic["Survived"].iloc[train]
    #生成规则
    alg.fit(train_predictors, train_target)
    #将规则运用于测试集
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)
    
    
import numpy as np

#将3个预测numpy数组连接为1个
predictions = np.concatenate(predictions, axis=0)

#生存预测大于0.5视为存活，反之视为死亡
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = 0
for i in range(len(predictions)):
    if predictions[i] == titanic["Survived"][i]:
        accuracy += 1

#查看预测的正确率
accuracy = accuracy / len(predictions)
print(accuracy)


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

#初始化算法
alg = LogisticRegression(random_state=1)
#计算所有交叉验证折叠的准确度得分
scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
#取分数的平均值。
print(scores.mean())


import pandas
#数据处理过程与上面一致
titanic_test = pandas.read_csv("D:/python代码/train.csv")

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

#查看票价的分布
plt.figure(figsize = (10,5))
titanic['Fare'].hist(bins = 70)
titanic.boxplot(column = 'Fare',by = 'Pclass',showfliers = False)
plt.show()

#查看票价与存活的关系
fare_not_survived = titanic['Fare'][titanic['Survived'] == 0]
fare_survived = titanic['Fare'][titanic['Survived'] == 1]
average_fare=pandas.DataFrame([fare_not_survived.mean(),fare_survived.mean()])
std_fare=pandas.DataFrame([fare_not_survived.std(),fare_survived.std()])

average_fare.plot(yerr = std_fare,kind = 'bar', legend = False)

plt.show()
#船票缺失值较少，采用均值填充
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())



#初始化算法
alg = LogisticRegression(random_state=1)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#利用训练数据训练算法
alg.fit(titanic[predictors], titanic["Survived"])

#使用测试集进行预测。
predictions = alg.predict(titanic_test[predictors])

#创建一个新的dataframe，并写入文件
submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("kaggle.csv", index=False)


from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#用默认参数初始化算法
#n_估计量是生成树的数量
#min_samples_split是进行拆分所需的最小行数
#min_samples_leaf是在树枝结束处（树的底部点）可以拥有的最小样本数
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

#计算所有交叉验证折叠的准确度得分
kf = model_selection.KFold(titanic.shape[0], n_splits=3, random_state=1)
scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

#取分数的平均值
print(scores.mean())



#对随机森林算法的参数进行调整
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
kf = model_selection.KFold(titanic.shape[0], n_splits=3, random_state=1)
scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

#打印平均分
print(scores.mean())


fig, ax = plt.subplots(1,2,figsize = (18,8))
titanic[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax = ax[0])
titanic[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax = ax[1])

#提取每个人的家庭情况和名字长度

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

titanic[['FamilySize','Survived']].groupby(['FamilySize']).mean().plot.bar()
titanic[['NameLength','Survived']].groupby(['NameLength']).mean().plot.bar()


import re

def get_title(name):
    #使用正则表达式搜索标题。标题总是由大写字母和小写字母组成，并以句点结尾。
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #如果标题存在，则提取并返回。
    if title_search:
        return title_search.group(1)
    return ""

#获取所有标题并打印每个标题出现的频率。
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

#将每个标题用一个整数替换。对于比较少见的标题，用其他标题代替
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v

#在标题列中添加。
titanic["Title"] = titles

titanic[['Title','Survived']].groupby(['Title']).mean().plot.bar()


import operator

#将姓氏放到到id的字典
family_id_mapping = {}


#获取给定行的id的函数
def get_family_id(row):
    #用逗号分开查找姓氏
    last_name = row["Name"].split(",")[0]
    #创建族id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    #在映射中查找id
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            #从映射中获取最大id，如果没有id，则添加一个
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


#使用apply方法获取族ID
family_ids = titanic.apply(get_family_id, axis=1)

#把ID在3个以下的放在一起
family_ids[titanic["FamilySize"] < 3] = -1

#打印每个唯一id的计数。
print(pandas.value_counts(family_ids))

titanic["FamilyId"] = family_ids


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId",
              "NameLength"]

#执行功能选择
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

#获取每个特征的原始p值，并将p值转换为分数
scores = -np.log10(selector.pvalues_)

#画出分数
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

#只选择四个最好的功能。
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
kf = model_selection.KFold(titanic.shape[0], n_splits=3, random_state=1)
scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# 打印分数
print(scores.mean())


from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

#对线性预测因子采取梯度提升分类器
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
     ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

#初始化交叉验证折叠
kf = KFold(titanic.shape[0], n_splits=3, random_state=1)

predictions = []
for train, test in kf.split(titanic):
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    #在每一个折叠上对每个算法进行预测
    for alg, predictors in algorithms:
        #在训练数据上拟合算法。
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        #选择并预测测试折叠。
        #将数据帧转换为所有float并避免sklearn错误所必需的。
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    #平均预测值代表最终的分类。
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    #任何大于.5的值都被视为存活，低于.5的值为被视为死亡。
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

#把所有的预测放在一个数组中。
predictions = np.concatenate(predictions, axis=0)

#通过与训练数据进行比较来计算精度。
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)


#向测试集中添加标题。
titles = titanic_test["Name"].apply(get_title)
#添加Dona标题
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2,
                 "Dona": 10}
for k, v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
#检查每个唯一标题的计数。
print(pandas.value_counts(titanic_test["Title"]))

#添加家庭人数特征
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids

titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))


predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    #利用完整的训练数据拟合算法
    alg.fit(titanic[predictors], titanic["Survived"])
    #使用测试数据集进行预测，将所有列转换为浮点
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:, 1]
    full_predictions.append(predictions)

#梯度提升分类器能产生更好的预测结果，因此将其加权得更高
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

for i in range(len(predictions)):
    if predictions[i] <= 0.5:
        predictions[i] = 0
    elif predictions[i] > 0.5:
        predictions[i] = 1

predictions = predictions.astype(int)

submission = pandas.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("my_submission.csv", index=False)
print('success')


