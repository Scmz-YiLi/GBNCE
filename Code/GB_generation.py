import numpy as np
import warnings
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')

def get_radius(gb):
    real_gb = gb[:, :-1]
    center = real_gb.mean(0)
    return max(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))

def spilt_ball_2(data):
    real_data = data[:, :-1]
    ball1 = []
    ball2 = []
    D = cdist(real_data, real_data)
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    for j in range(len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    return [np.array(ball1), np.array(ball2)]

def get_density_volume(gb):
    real_gb = gb[:, :-1]
    num = len(real_gb)
    center = real_gb.mean(0)
    sum_radius = np.sum(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))
    mean_radius = sum_radius / num
    if mean_radius != 0:
        return num / sum_radius
    return num

def division_ball(gb_list):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >= 4:
            ball_1, ball_2 = spilt_ball_2(gb)
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb)
                continue
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = (w1 * density_child_1 + w2 * density_child_2)
            if w_child > density_parent:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    return gb_list_new

def normalized_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if len(gb) < 2:
            gb_list_temp.append(gb)
        else:
            ball_1, ball_2 = spilt_ball_2(gb)
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_1, ball_2])
    return gb_list_temp

def get_GB(data):
    data_num = data.shape[0]
    index = np.arange(data_num).reshape(data_num, 1)
    data = np.hstack((data, index))
    gb_list_temp = [data]
    while True:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_ball(gb_list_temp)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
            
    radius = [get_radius(gb) for gb in gb_list_temp if len(gb) >= 2]
    radius_detect = max(np.median(radius), np.mean(radius))
    
    while True:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(gb_list_temp, radius_detect)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
            
    return gb_list_temp
