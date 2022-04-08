import src.Preprocessing
import pandas as pd

if __name__ == '__main__':
    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
    ready_df = preproc.run()
    ready_df.to_csv('./data/preprocessed_data_v2.csv', index=False)


