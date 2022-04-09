import src.Preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    prep_df = pd.read_csv("./data/preprocessed_data.csv", on_bad_lines="skip")

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = prep_df.select_dtypes(include=numerics)

    # y = prep_df["FULLVAL"]
    # x = prep_df["AVTOT"]

    df_tcA = numeric_df.loc[(prep_df['TAXCLASS'] == "1") &
                         (prep_df['TAXCLASS'] == "1A") &
                         (prep_df['TAXCLASS'] == "1B") &
                         (prep_df['TAXCLASS'] == "1C") &
                         (prep_df['TAXCLASS'] == "2A") &
                         (prep_df['TAXCLASS'] == "2B") &
                         (prep_df['TAXCLASS'] == "2C")
    ]

    df_tcB = numeric_df.loc[(prep_df['TAXCLASS'] == "2") &
                         (prep_df['TAXCLASS'] == "3") &
                         (prep_df['TAXCLASS'] == "4")
    ]

    df_tc1A = numeric_df.loc[(prep_df['TAXCLASS'] == "1A") # &
                             # (prep_df['BORO'] == 4)
    ]

    y = df_tc1A["FULLVAL"]
    x = df_tc1A["AVTOT"]
    plt.scatter(x, y)
    plt.show()
