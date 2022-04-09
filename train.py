import src.Preprocessing
import src.TTrain
import json

if __name__ == '__main__':

    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", True)
    f = open("src/conf_train.json")
    conf = json.load(f)
    for tax, tax_conf in conf['tax'].items() :
        #print(tax_conf)
        category = tax_conf['category']
        if tax_conf['boro'] :
            for boro, boro_conf in tax_conf['items'].items():
                preproc.refresh_data( tax_conf['boro'])
                boro_num = int(boro)
                X, Y = preproc.run(category, boro_conf['columns'],boro)
                print(X)
                print(Y)
                path = 'models/' + 'model_' + tax + '_' + boro + '.sav'
                train = src.TTrain.TTrain(boro_conf['model_name'], X,Y, path )
                train.train_linear_regression()
        else :
            preproc.refresh_data( tax_conf['boro'])
            X, Y = preproc.run(category, boro_conf['columns'])
            print(X)
            print(Y)
            path = 'models/' + 'model_' + tax + '.sav'
            train = src.TTrain.TTrain(boro_conf['model_name'], X,Y, path )
            train.train_linear_regression()

