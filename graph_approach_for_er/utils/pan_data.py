import pandas as pd
from sklearn.model_selection import train_test_split


class PanDataPreprocessing:
    def __init__(self, path_to_data: str):
        self.labels = pd.read_json(path_to_data, lines=True)
        self.data = pd.read_json(path_to_data.replace("-truth", ""), lines=True)

        df = pd.merge(
            left=self.labels,
            right=self.data,
            on="id"
        )

        df["left"] = df.apply(lambda x: str(x["pair"][0]), axis=1)
        df["right"] = df.apply(lambda x: str(x["pair"][1]), axis=1)
        df["label"] = df.apply(lambda x: 1 if True else 0, axis=1)
        self.relevant_columns = ["left", "right", "label"]

        self.df_train, df_other = train_test_split(df, random_state=123, test_size=0.2)
        self.df_val, self.df_test = train_test_split(df_other, random_state=123, test_size=0.5)

    def get_df_train(self):
        return self.df_train[self.relevant_columns].copy(deep=True).reset_index(drop=True)

    def get_df_val(self):
        return self.df_val[self.relevant_columns].copy(deep=True).reset_index(drop=True)

    def get_df_test(self):
        return self.df_test[self.relevant_columns].copy(deep=True).reset_index(drop=True)
