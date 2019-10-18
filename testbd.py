import numpy as np 
import pandas as pd 


face_coords = pd.read_csv('./face_coors.csv')
face_coords['coords'] = None

face_rects = pd.read_csv('./face_rects.csv')

print(face_coords.head())
print(face_rects.head())

root_dir = '../alfw/flickr/'
current_id = 0
current_path = ""
current_coords = {}
current_rect = []
face_coords_dic = {}
face_num = 0
face_list_df = pd.DataFrame(columns=('file_path', 'face_coords', 'face_rect'))
print(face_list_df.head())
for index, row in face_coords.iterrows():
    
    if current_id == 0:
        current_id = row['face_id']
        current_path = row['file_path']
    
    print(index, "/", face_coords.shape[0])

    # face_list_df.loc[index] = {'file_path': root_dir+row['file_path'], 'face_coords': None}

    if current_id != row['face_id']:
        for index_r, row_r in face_rects.iterrows():
            if current_id == row_r['face_id']:
                current_rect = [row_r['x'], row_r['y'], row_r['w'], row_r['h']]
                break
        new_face = True
        face_list_df.loc[face_num] = {'file_path': root_dir+current_path,
                                      'face_coords': current_coords.copy(),
                                      'face_rect': current_rect.copy()}
        face_num += 1
        # face_coords_dic[current_id] = current_coords.copy()
        current_coords.clear()
        current_coords[row['feature_id']] = [row['x'], row['y']]
        current_id = row['face_id']
        current_path = row['file_path']
    else:
        new_face = False
        current_coords[row['feature_id']] = [row['x'], row['y']]

    # if face_num > 5:
    #     break

# print(face_list_df.head())

face_list_df.to_csv('./face_list_df_with_rect.csv', index=False)
    



