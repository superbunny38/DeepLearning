import pandas as pd
'''
Create dataframe with voxel locations and corresponding value from a 3d image array
'''
def create_3d_df(array3d):
    arr_x = []
    arr_y = []
    arr_z = []
    arr_val = []
    for x_idx in range(array3d.shape[0]):
        for y_idx in range(array3d.shape[1]):
            for z_idx in range(array3d.shape[2]):
                arr_x.append(x_idx)
                arr_y.append(y_idx)
                arr_z.append(z_idx)
                arr_val.append(array3d[x_idx,y_idx,z_idx])
    df = pd.DataFrame({'x':arr_x,'y':arr_y,'z':arr_z,'val':arr_val})
    return df