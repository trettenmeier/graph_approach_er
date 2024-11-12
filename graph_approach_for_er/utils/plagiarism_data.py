import pandas as pd
from sklearn.model_selection import train_test_split


class PlagiarismDataPreprocessing:
    def __init__(self, path_to_data: str):
        df = pd.read_table(path_to_data, names=["left", "right", "label"])

        print("sampling 10000 rows")
        df = df.sample(n=10000, random_state=123123).reset_index(drop=True)

        df["left"] = df["left"].astype(str)
        df["right"] = df["right"].astype(str)
        df["label"] = df["label"].astype(int)
    

        self.df_train, df_other = train_test_split(df, random_state=123, test_size=0.2)
        self.df_val, self.df_test = train_test_split(df_other, random_state=123, test_size=0.5)

    def get_df_train(self):
        return self.df_train.copy(deep=True).reset_index(drop=True)

    def get_df_val(self):
        return self.df_val.copy(deep=True).reset_index(drop=True)

    def get_df_test(self):
        return self.df_test.copy(deep=True).reset_index(drop=True)
