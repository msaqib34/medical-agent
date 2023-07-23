import pandas

def merge_columns(row):
    return row.tolist()

precaution_df = pandas.read_csv("data/symptom_precaution.csv")
columns_to_merge = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
precaution_df["Precautions"] = precaution_df[[columns_to_merge]].apply(merge_columns, axis=1)
precaution_df.drop(columns=columns_to_merge, inplace=True)
precaution_df.head()
precaution_df.to_csv("symptom_precaution_modified.csv", index=False)

dataset_df = pandas.read_csv("data/dataset.csv")
columns_to_merge = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5", "Symptom_6", "Symptom_7", "Symptom_8", "Symptom_9", "Symptom_10", "Symptom_11", "Symptom_12", "Symptom_13", "Symptom_14", "Symptom_15", "Symptom_16", "Symptom_17"]
dataset_df["Symptoms"] = dataset_df[columns_to_merge].apply(merge_columns, axis=1)
dataset_df.drop(columns=columns_to_merge, inplace=True)
dataset_df.head()
dataset_df.to_csv("dataset_modified.csv", index=False)