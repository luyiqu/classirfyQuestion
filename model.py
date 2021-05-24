#coding:utf-8
import jieba.analyse
import jieba
import jieba.posseg
import os
import pandas as pd
import re
import numpy as np
import time
import word2vec
import gensim
import multiprocessing
from gensim.models import KeyedVectors,word2vec,Word2Vec
from sklearn.multioutput import MultiOutputClassifier 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
from sklearn.metrics import f1_score, precision_score, recall_score



r1 = "[a-zA-Z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、：；;《》“”~@#￥%……&*（）]+"
r2list = "=}-<>\∑↑:≤×<→><`‘ ’'[ ]{}()-」「|∏Σ∈≡÷＞≥＝"
posseg_flag = ['Ag','a','ad','d','dg','nr','ns','t','y']

def stopwordslist():
    stopwords = [line.strip() for line in open('./中文停用词表.txt',encoding='UTF-8').readlines()]
    return stopwords

def splitwordforHAN(content):
    stopwords = stopwordslist()
    result = []
    sents = re.split('(。|，|；|！|\!|\.|？|\?)',content)
    outstr = []
    for sent in sents:
        sent = re.sub(r1,"",sent)
        sent = myresub(sent)
        sent_depart = jieba.posseg.cut(sent)
        for word in sent_depart:
            if(word.word not in stopwords and word.flag not in posseg_flag):
                outstr.append(word.word)
    
    return outstr



def myresub(content):   #去除一些r2list中的
    outstr = ''
    for con in content:
        if con not in r2list:
            outstr += con
    return outstr    

#加载Word2vec模型
def LoadWord2Vec(word2vec_path):
    w2vModel = Word2Vec.load(word2vec_path)
    return w2vModel    

def getLabel(question):
    model_path = "./model/word2vec_delete_a_18.model"
    totalTotal = ["基础","模拟","贪心","查找","排序","递归","分治","搜索","动态规划","数学","数组","链表","栈","队列","字符串","图","树","哈希"]
    labelResult = []
    splitword = splitwordforHAN(question) #分词
    #得到词向量
    model=LoadWord2Vec(model_path)
    test_vector = sentence_vec(splitword,model)
    classiferML = loadTree("./model/cls")
    test_y = classiferML.predict([test_vector])
    print(test_y)
    for i in range(test_y[0].size):
        if(test_y[0][i]==1):
            
            labelResult.append(totalTotal[i])

    return labelResult    

def loadTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

# 得到每个句子的句向量(根据每个句子的词向量求和取平均）
def sentence_vec(sentence,model):
    strlen = len(sentence)
    sum = np.zeros_like(model.wv['发现'])
#         print(sum)
    for word in sentence:
        sum += model.wv[word]    
    return sum/strlen    


## 目前只适配了一道题目的，只需要调用getLabel即可    

question =' 平时的练习和考试中，我们经常会碰上这样的题：命题人给出一个例句，要我们类比着写句子。这种往往被称为仿写的题，不单单出现在小学生的考试中，也有时会出现在中考中。许多同学都喜欢做这种题，因为较其它题显得有趣。仿写的句子往往具有“A__B__C”的形式，其中A，B，C是给定的由一个或多个单词组成的短句，空的部分需要学生填写。当然，考试的时候空在那里也是可以的。例如，“其实天不暗阴云终要散，其实 ，其实 ，其实路不远一切会如愿，艰难困苦的日子里我为你祈祷，请你保重每一天”。再比如，“见了大海的汹涌，没见过大山的巍峨，真是遗憾；见了大山的巍峨，没见过 ，还是遗憾。出发吧，永远出发。 ，人有不老的心情。”\r\n由于现在是网络时代，我们不再只能仿写命题人命的题，我们可以仿写网上各种句子和段落。2011年3月26日，某人在博客上发布了的消息就惹来了很多人的仿写。\r\n很难过吧。。。考得完爆了。。。\r\n。。。。。。其实也没什么可以说的。。。都是蒟蒻的借口罢了。。。\r\n。。。自己果然还只是半吊子水平呢。。。。\r\n。。。祝大家都能进省队。。。其实只要不要有遗憾就好了呢。。。\r\n虽然我很遗憾或许不能走下去了。。。。。\r\n886\r\n在网络上广泛流传的仿写，因为在某些地方有独到之处，大都被命名为“某某体”。打开人人，刷新微博，你也能发现这样和那样的体，比如，对不起体，**说明他爱你体等等。金先生注意到了这一现象，他敏锐地认为这是一个很有价值的研究课题，于是就其展开研究，打算发一篇paper。由于在网上发消息，人们有了更大的灵活度，人们有时因为表达的需要，还往原本固定的A, B, C中添加一些修饰的词语。这就给辨别一个句子或段落是否是另一个句子或段落的仿写增加了困难。\r\n金先生现在研究一种形如“A*B*C”的体作品，其中A, B, C分别是某个由若干单词组成的短句，*代表0个或多个单词。他在网上找了大量的体作品，不过很多体作品不太合乎原作者的格式，也就是相当于在正规的体作品中插入了0个或多个单词。\r\n由于数据量太大，金先生无法一个一个看过去，于是想请你帮忙，去掉尽量少的单词，使它成为指定的体。\r\n '
labels = getLabel(question)
print(labels)