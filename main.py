import os
import src.Preprocessing
import src.Train
import json
import numpy as np
from pathlib import Path


def grid_search_columns_thresh():
    f = open("src/conf.json")
    conf = json.load(f)
    print(conf)
    model = conf['model']


    required_columns = [
        # {"cols": ['Longitude'], "name": 'long'},
        # {"cols": ['Latitude'], "name": 'lat'},
        # {"cols": ['Latitude', 'Longitude'], "name": 'lat_long'},
        # {"cols": ['Latitude', 'Longitude', 'STORIES'], "name": 'lat_long_stories'},
        # {"cols": ['Latitude', 'STORIES'], "name": 'lat_stories'},
        # {"cols": ['Longitude', 'STORIES'], "name": 'long_stories'},
        # {"cols": ['Latitude', 'Longitude', 'STORIES', 'BLDAREA', 'LTAREA'], "name": 'lat_long_stories_bldarea_ltarea'},

    ]
    thresholds = [0.5]

    for req_columns in required_columns:
        for threshold in thresholds:
            dir_path = Path('results/grid_search') / f'{req_columns["name"]}_{threshold}'
            os.mkdir(dir_path)

            total_results = []
            for tax, tax_conf in conf['tax'].items() :
                #print(tax_conf)
                category = tax_conf['category']
                if tax_conf['boro'] :
                    for boro, boro_conf in tax_conf['items'].items():
                        preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", tax_conf['boro'], required_columns=req_columns['cols'])
                        # threshold = boro_conf['threshold']
                        boro_num = int(boro)
                        X, Y = preproc.run(threshold, category, boro)
                        print(X)
                        print(Y)
                        train = src.Train.Train(X,Y, model)
                        res, results_err = train.evaluate_regression()
                        total_results.append(results_err)
                        name = dir_path / ('model_' + tax + '_' + boro + '_' + str(threshold) + '.json')
                        with open(name, 'w') as f:
                            json.dump(res, f, indent=2)
                else :
                    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", tax_conf['boro'], required_columns=req_columns['cols'])
                    # threshold = tax_conf['threshold']
                    X, Y = preproc.run(threshold, category)
                    print(X)
                    print(Y)
                    train = src.Train.Train(X,Y, model)
                    res, results_err = train.evaluate_regression()
                    total_results.append(results_err)
                    name = dir_path / ('model_' + str(model) + '_' + tax + '_' + str(threshold) + '.json')
                    with open(name, 'w') as f:
                        json.dump(res, f, indent=2)

            results = { k: [] for k in list(total_results[0].keys())}
            for res in total_results:
                for k, v in res.items():
                    results[k].append(v)

            mean_std_vals = {}
            for k in results.keys():
                if k != 'len':
                    mean_std_vals[f'{k}_avg'] = np.average(results[k], weights=results['len'])
            results = { **results, **mean_std_vals }

            name = dir_path / ('model_total_' + str(threshold) + '.json')
            with open(name, 'w') as f:
                json.dump(results, f, indent=2)

if __name__ == '__main__':

    # grid_search_columns_thresh()


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

    dir_path = Path('results')

    total_results = []
    for tax, tax_conf in conf['tax'].items() :
        #print(tax_conf)
        category = tax_conf['category']
        if tax_conf['boro'] :
            for boro, boro_conf in tax_conf['items'].items():
                preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", tax_conf['boro'])
                threshold = boro_conf['threshold']
                boro_num = int(boro)
                X, Y = preproc.run(threshold, category, boro)
                print(X)
                print(Y)
                train = src.Train.Train(X,Y, model)
                res, results_err = train.evaluate_regression()
                total_results.append(results_err)
                name = dir_path / ('model_' + str(model) + tax + '_' + boro + '_' + str(threshold) + '.json')
                with open(name, 'w') as f:
                    json.dump(res, f, indent=2)
        else :
            preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", tax_conf['boro'])
            threshold = tax_conf['threshold']
            X, Y = preproc.run(threshold, category)
            print(X)
            print(Y)
            train = src.Train.Train(X,Y, model)
            res, results_err = train.evaluate_regression()
            total_results.append(results_err)
            name = dir_path / ('model_' + str(model) + '_' + tax + '_' + str(threshold) + '.json')
            with open(name, 'w') as f:
                json.dump(res, f, indent=2)

    results = { k: [] for k in list(total_results[0].keys())}
    for res in total_results:
        for k, v in res.items():
            results[k].append(v)

    mean_std_vals = {}
    for k in results.keys():
        if k != 'len':
            mean_std_vals[f'{k}_avg'] = np.average(results[k], weights=results['len'])
    results = { **results, **mean_std_vals }

    name = dir_path / ('model_total_' + str(threshold) + '.json')
    with open(name, 'w') as f:
        json.dump(results, f, indent=2)







