# -*- coding: utf-8 -*-
import graphlab, jieba
import re, chardet, codecs, sys

def train_model(train_data):
    # data = {'feature':[],'label':[],'sentence':[]}
    svm_model = graphlab.svm_classifier.create(train_data, target='label', features=['feature'], verbose=False)
    return svm_model

def read_train_data():
    data = graphlab.SFrame.read_csv('./data/train.txt', delimiter='\t', header=False, verbose=False)
    data = graphlab.cross_validation.shuffle(data)
    new_data = graphlab.SFrame()
    new_data['feature'] = graphlab.text_analytics.count_words(data['X2'])
    new_data['label'] = data['X1']
    new_data['sentence'] = data['X2']
    return new_data
    
def read_data():
    # 修改small_test.txt to test.txt，训练完整
    data = graphlab.SFrame.read_csv('./small_test.txt', delimiter='\t', header=False, verbose=False)
    data['feature'] = graphlab.text_analytics.count_words(data['X1'])
    return data

def generate_new_data(test_data, model):
    test_data['label'] = model.predict(test_data, output_type='margin')
    new_test_data = graphlab.SFrame()
    new_test_data['feature'] = test_data['feature']
    new_test_data['label'] = test_data['label']
    new_test_data['sentence'] = test_data['X1']
    sample_data_1 = new_test_data.sort('label', ascending = False)[:500]
    sample_data_0 = new_test_data.sort('label', ascending = False)[-500:]
    sample_data_1['label'] = sample_data_1['label'].apply(lambda x:int(1))
    sample_data_0['label'] = sample_data_0['label'].apply(lambda x:int(0))
    # generate test_data for next iter
    sample_data_unknown = new_test_data.sort('label', ascending = False)[500:-500]
    sample_data_unknown = sample_data_unknown.rename({'sentence':'X1'})
    sample_data_unknown.remove_column('label')
    return sample_data_1, sample_data_0, sample_data_unknown


if __name__ == "__main__":
    train_data = read_train_data()
    test_data = read_data()
    n = 0
    while n < 4: # 修改迭代次数4
        print "train_data", train_data.num_rows()
        print "test_data", test_data.num_rows()
        print "iter", n, "-"*10 
        model = train_model(train_data)
        _1, _2, _3 = generate_new_data(test_data, model)
        new_train_data = train_data.append(_1).append(_2)
        test_data = _3
        train_data = new_train_data
        n += 1
    op = open('final_data.txt', 'w')
    for l,s in zip(train_data['label'], train_data['sentence']):
        op.write(str(l)+"\t"+s+"\n")
    op.close()
        
