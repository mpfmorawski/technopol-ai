import src.Preprocessing
import src.Pred
import json
import sklearn.metrics as metrics
import numpy as np

if __name__ == '__main__':

    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv", True)
    f = open("src/conf_train.json")
    conf = json.load(f)
    pred_total = []
    actual_total = []
    x_total = []
    for tax, tax_conf in conf['tax'].items() :
        #print(tax_conf)
        category = tax_conf['category']
        if tax_conf['boro'] :
            for boro, boro_conf in tax_conf['items'].items():
                preproc.refresh_data( tax_conf['boro'])
                boro_num = int(boro)
                X, Y = preproc.run(category, boro_conf['columns'],boro)
                path = 'models/' + 'model_' + tax + '_' + boro + '.sav'
                pred = src.Pred.Pred(boro_conf['model_name'], X,Y, path )
                pred, actuals = pred.pred_linear_regression()
                _mae = metrics.mean_absolute_error(actuals, pred)
                _r2 = metrics.r2_score(actuals, pred)
                mean = np.mean(actuals)
                mea_ref = _mae/mean
                print(f"MAE: {_mae} \nR2: {_r2} \n Mean: {mean}\n Mean_ref: {mea_ref}")
                pred_total.extend(pred)
                actual_total.extend(actuals)
                
        else :
            preproc.refresh_data( tax_conf['boro'])
            X, Y = preproc.run(category, boro_conf['columns'])
            path = 'models/' + 'model_' + tax + '.sav'
            pred = src.Pred.Pred(boro_conf['model_name'], X,Y, path )
            pred, actuals = pred.pred_linear_regression()
            _mae = metrics.mean_absolute_error(actuals, pred)
            _r2 = metrics.r2_score(actuals, pred)
            mean = np.mean(actuals)
            mea_ref = _mae/mean
            print(f"MAE: {_mae} \nR2: {_r2} \n Mean: {mean}\n Mean_ref: {mea_ref}")
            pred_total.extend(pred)
            actual_total.extend(actuals)

    print(len(actual_total))
    mae = metrics.mean_absolute_error(actual_total, pred_total)
    r2 = metrics.r2_score(actual_total, pred_total)

    mean_total = np.mean(actual_total)
    mea_ref_total = mae/mean
    print(f"MAE: {mae} \nR2: {r2} \n Mean: {mean_total}\n Mean_ref: {mea_ref_total}")