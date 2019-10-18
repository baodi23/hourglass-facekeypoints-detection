import pandas as pd

df_rects = pd.read_csv('face_rects.csv')
print(df_rects.head())

df_files = pd.read_csv('face_list_df.csv')
print(df_files.head())

df_coors = pd.read_csv('face_coors.csv')
print(df_coors.head())