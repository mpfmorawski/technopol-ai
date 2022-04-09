import src.Preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.Train

if __name__ == '__main__':
    prep_df = pd.read_csv("./data/preprocessed_data.csv", on_bad_lines="skip")

    #category "2,3,4"
    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
    treshold = 1
    category = ['1']
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


    # y = prep_df["FULLVAL"]
    # x = prep_df["AVTOT"]

    # df_tcA = numeric_df.loc[(prep_df['TAXCLASS'] == "1") &
    #                      (prep_df['TAXCLASS'] == "1A") &
    #                      (prep_df['TAXCLASS'] == "1B") &
    #                      (prep_df['TAXCLASS'] == "1C") &
    #                      (prep_df['TAXCLASS'] == "2A") &
    #                      (prep_df['TAXCLASS'] == "2B") &
    #                      (prep_df['TAXCLASS'] == "2C")
    # ]
    #
    # df_tcB = numeric_df.loc[(prep_df['TAXCLASS'] == "2") &
    #                      (prep_df['TAXCLASS'] == "3") &
    #                      (prep_df['TAXCLASS'] == "4")
    # ]
    #
    # df_tc1A = numeric_df.loc[(prep_df['TAXCLASS'] == "1A") # &
    #                          # (prep_df['BORO'] == 4)
    # ]
    #
    # y = df_tc1A["FULLVAL"]
    # x = df_tc1A["AVTOT"]
    # plt.scatter(x, y)
    # plt.show()
