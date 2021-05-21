from feature_extraction import CrossFeatures
from sklearn.datasets import load_diabetes, load_breast_cancer

import IPython
def display(*dfs):
    for df in dfs:
        IPython.display.display(df)


df = load_diabetes(as_frame=True)
# df = load_breast_cancer(as_frame=True)
df = df.frame
# print(df.nunique())


cat_cols = [f for f in df.columns if df[f].nunique()<60]
float_cols = df.drop(cat_cols, axis=1).columns.values.tolist()
print(cat_cols, float_cols)

cr_f = CrossFeatures(cat_cols=cat_cols,
                     float_cols=float_cols)
cr_f.fit(df)
# cr_f._gen_cols(df)
# display(df.columns)

cr_f.transform(df, total=False)









