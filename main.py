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
    model = conf['model']
    for tax, tax_conf in conf['tax'].items() :
        if tax == '234' :
            category = ['2','3', '4']
        elif tax == '1B-1C':
            category = ['1B', '1C']
        else : category = [tax]
        if tax_conf['boro'] :
            for boro, boro_conf in tax_conf['items'].items():
                preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", tax_conf['boro'])
                threshold = boro_conf['threshold']
                boro_num = int(boro)
                X, Y = preproc.run(threshold, category, boro)
                print(X)
                print(Y)
                train = src.Train.Train(X,Y, model)
                res = train.evaluate_regression()
                name = 'results/' + 'model_' + tax + '_' + boro + '_' + str(threshold) + '.json'
                with open(name, 'w') as f:
                    json.dump(res, f)
        else :
            preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", tax_conf['boro'])
            threshold = tax_conf['threshold']
            X, Y = preproc.run(threshold, category)
            print(X)
            print(Y)
            train = src.Train.Train(X,Y, model)
            res = train.evaluate_regression()
            name = 'results/' +'model_' + str(model) + '_' + tax + '_' + str(threshold) + '.json'
            with open(name, 'w') as f:
                json.dump(res, f)






