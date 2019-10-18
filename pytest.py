import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

df_file = pd.read_csv('./face_list_df_with_rect.csv')
print(df_file.shape)
print(df_file.head())

for index, row in df_file.iterrows():
    image_path = df_file.loc[index]['file_path']
    df_file.loc[index]['file_path'] = image_path.replace('..', '/data')

df_file.to_csv('cdl_data.csv')
#
    # df_train = df_file[:23385]
    # print(df_train.shape)
    # df_test = df_file[23385:]
    # print(df_test.shape)
    # df_train.to_csv('./face_train_with_rect.csv')
    # df_test.to_csv('./face_test_with_rect.csv')

