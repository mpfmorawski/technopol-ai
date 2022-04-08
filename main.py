import src.Preprocessing

if __name__ == '__main__':
    preproc = src.Preprocessing.Preprocessing("./data/train_data.csv")
    ready_df = preproc.run()
    print(ready_df)