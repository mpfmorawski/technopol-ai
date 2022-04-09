import src.Preprocessing
import src.Train
import json

if __name__ == '__main__':
    #category "2,3,4"
    # preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
    # treshold = 1
    # category = ['2','3','4']
    # X, Y = preproc.run(treshold, category)
    # print(X)
    # print(Y)
    # train = src.Train.Train(X,Y)
    # train.evaluate_linear_regression()

    #analyze category 1
    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
    # treshold = 0.7
    # category = ['2A','2B','2C']
    # boro = 2
    # X, Y = preproc.run(treshold, category, boro)
    # print(X)
    # print(Y)
    # train = src.Train.Train(X,Y)
    # train.evaluate_linear_regression()
    f = open("src/conf.json")
    conf = json.load(f)
    print(conf)

    for tax, tax_conf in conf['tax'].iteritems() :
        if tax != '234' :
            category = [tax]
        else : category = ['2','3', '4']
        if tax_conf['boro'] :
            for boro, boro_conf in tax_conf['items']:
                preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
                threshold = boro_conf['threshold']
                boro_num = int(boro)


    


