import src.Preprocessing
import src.Train

if __name__ == '__main__':
    #category "2,3,4"
    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
    treshold = 1
    category = ['2','3','4']
    X, Y = preproc.run(treshold, category)
    print(X)
    print(Y)
    train = src.Train.Train(X,Y)
    train.evaluate_linear_regression()

    #analyze category 1
    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
    treshold = 0.8
    category = ['1']
    X, Y = preproc.run(treshold, category)
    print(X)
    print(Y)

