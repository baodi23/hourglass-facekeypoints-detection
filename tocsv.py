import pandas as pd 
import sqlite3 

with sqlite3.connect('../alfw/aflw.sqlite') as con:

    # df_faces  = pd.read_sql_query("SELECT * FROM Faces", con=con).to_csv('./faces.csv')
    # df_coors  = pd.read_sql_query("SELECT * FROM FeatureCoords", con=con).to_csv('./face_coors.csv')
    # df_images = pd.read_sql_query("SELECT * FROM FaceImages", con=con).to_csv('./face_images.csv')
    df_rects = pd.read_sql_query("SELECT * FROM FaceRect", con=con).to_csv('./face_rects.csv')
