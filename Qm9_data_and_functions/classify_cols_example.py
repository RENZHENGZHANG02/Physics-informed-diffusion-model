import pandas as pd

df = pd.read_csv("heat_related_output.csv")

def classify_gap(gap):
    if gap < 0.22:
        return 0
    elif gap < 0.28:
        return 1
    else:
        return 2

df["gap_class"] = df["gap"].apply(classify_gap)

df[["smiles", "Cv", "gap_class"]].sample(n=2000, random_state=42).to_csv("classification_dataset.csv", index=False)
