import pandas as pd
import numpy as np
import os

ACC_THRESHOLD = 40
WIN_SIZE_BEFORE = 1 # # of seconds before first 40-speak
WIN_SIZE_AFTER = 1 # of seconds after first 40-speak
WIN_LEN = 1000

def compute_features_per_window(window, window_norms):
	# print(window_norms)
	w_max = window_norms.max()
	w_mean = window_norms.mean()
	w_std = window_norms.std()
	w_rms = np.sqrt(np.square(window_norms).sum())

	w_nonans = window[np.isfinite(window['norm_after'])]
	w_arc_len = (w_nonans['norm_after'] - w_nonans['norm']).abs().sum() / w_nonans['norm'].size

	w_arc_len_std = w_arc_len * w_std
	w_mean_abs = window_norms.abs().mean()

	return w_max, w_mean, w_std, w_rms, w_arc_len, w_arc_len_std, w_mean_abs

def computer_features_per_file(file_name, class_label):

    print(file_name)

    # 1. read csv
    acc = pd.read_csv(file_name, names=['timestamp', 'x', 'y', 'z'], dtype={'timestamp':np.float64, 'x':np.float64, 'y':np.float64, 'z':np.float64}, index_col=False)
    # print(acc.head(5))

    # acc['timestamp'] = pd.to_numeric(acc['timestamp'], errors='coerce')
    # acc['x'] = pd.to_numeric(acc['x'], errors='coerce')
    # acc['y'] = pd.to_numeric(acc['y'], errors='coerce')
    # acc['z'] = pd.to_numeric(acc['z'], errors='coerce')

    # acc = acc[(np.isfinite(acc['x'])) & (np.isfinite(acc['y'])) & (np.isfinite(acc['z']))]
    acc = acc[(pd.notnull(acc['x'])) & (pd.notnull(acc['z'])) & (pd.notnull(acc['z']))]

    # 2. calculate norm sqrt(sum(x^2, y^2, z^2))
    xyz = acc[['x', 'y', 'z']]
    acc['norm'] = np.sqrt(np.square(xyz).sum(axis=1))
    acc['norm_before'] = acc['norm'].shift(1)
    acc['norm_after'] = acc['norm'].shift(-1)

    # print(acc.head(5))
    # print("row with timestamp 703090064: ", acc[(acc['timestamp'] == 703090064)])

    # 3. find all times, t, when value is ~40, i.e. find all crossings.
    norm_exceed = acc[(acc['norm_before'] < ACC_THRESHOLD) & (acc['norm'] >= ACC_THRESHOLD)]

    # print(norm_exceed.head(5))
    # norm_exceed = acc[(acc['norm'] < 40) & (acc['norm_after'] > 40) | 
    #                   (acc['norm'] > 40) & (acc['norm_after'] < 40)]
    # norm_exceed = acc[(acc['norm_before'] < 40) & (acc['norm_after'] > 40) | 
    #                   (acc['norm_before'] > 40) & (acc['norm_after'] < 40)]

    # 4. get 1s window before & after t
    time_crossing = norm_exceed['timestamp']

    # due to pos experiment noise, we crop times norm cross 40 manullys
    if class_label == '1':
        if '81452402' in file_name:
            time_exceed = np.array([161330.0, 216094.0, 287314.0, 353417.0, 416721.0, 475336.0, 535971.0, 596367.0, 656234.0, 726406.0, 
                                    785748.0, 848996.0, 909128.0, 969941.0, 1028077.0, 1092756.0, 1148382.0, 1211740.0, 1269078.0, 1329543.0,
                                    1967731.0, 2048085.0, 2136125.0, 2221992.0, 2298554.0, 2372050.0, 2444078.0, 2514334.0, 2592409.0, 2671525.0])
        elif 'e5b921a6' in file_name: 
            time_exceed = np.array([933876.0, 966708.0, 1013558.0, 1061703.0, 1101594.0, 1151936.0, 1195055.0, 1247129.0, 1292453.0, 1361568.0,
                                    1432105.0, 1500765.0, 1544397.0, 1595380.0, 1637371.0, 1697678.0, 1758845.0, 1822171.0, 1893906.0, 1951704.0,
                                    2018429.0, 2088872.0, 2128050.0, 2172948.0, 2223263.0, 2284761.0, 2342263.0, 2389745.0, 2439722.0, 2483499.0])
        else:
            print('unexpected pos file')
        # print(pd.Series(time_exceed).isin(time_crossing.tolist()))
    else:
        time_exceed = time_crossing


    # old featurizer: different for pos and neg datasets.
    for t in time_exceed:
        window_start = t - WIN_SIZE_BEFORE * WIN_LEN
        window_end = t + WIN_SIZE_AFTER * WIN_LEN
        
        window1 = acc[(acc['timestamp'] >= window_start) & (acc['timestamp'] < t)]
        window1_norms = window1['norm']
        window2 = acc[(acc['timestamp'] >= t) & (acc['timestamp'] <= window_end)]
        window2_norms = window2['norm']

        if window1.shape[0] > 1 and window2.shape[0] > 1:
            # 5. Compute features
            window1_max, window1_mean, window1_std, window1_rms, window1_arc_len, window1_arc_len_std, window1_mean_abs \
            = compute_features_per_window(window1, window1_norms)
            # 6. generate before window
            w1 = pd.Series([window1_max, window1_mean, window1_std, window1_rms, window1_arc_len, window1_arc_len_std, window1_mean_abs])

            # 5. Compute features
            window2_max, window2_mean, window2_std, window2_rms, window2_arc_len, window2_arc_len_std, window2_mean_abs \
            = compute_features_per_window(window2, window2_norms)
            # 6. generate after window
            w2 = pd.Series([window2_max, window2_mean, window2_std, window2_rms, window2_arc_len, window2_arc_len_std, window2_mean_abs])

            # 7. concatenate before and after windows
            label = pd.Series([class_label])
            data_point = pd.concat([label, w1, w2])
            dp[file_name+'_'+str(t)] = data_point

            # print(dp.T.shape)

# build a design matrix 
dp = pd.DataFrame()
# dp = pd.DataFrame(columns=('filename_time', 'class_label', 
                           # 'win1_max', 'win1_mean', 'win1_std', 'win1_rms', 'win1_arc_len', 'win1_arc_len_std', 'win1_mean_abs',
                           # 'win2_max', 'win2_mean', 'win2_std', 'win2_rms', 'win2_arc_len', 'win2_arc_len_std', 'win2_mean_abs')) 
# pre-allocate # of rows. keep a counter

if __name__ == '__main__':

    pos_path = 'data/theft_classifier_data/theft_CSVs'
    pos_csv_files = [f for f in os.listdir(pos_path) if f.endswith('.csv') and 'BatchedAccelerometer' in f]
    for f in pos_csv_files:
        computer_features_per_file(os.path.join(pos_path, f), '1')

    # neg_path = 'data/theft_classifier_data/neg_CSVs'
    neg_path = 'data/theft_classifier_data/sensor_research'
    neg_csv_files = [f for f in os.listdir(neg_path) if f.endswith('.csv') and 'BatchedAccelerometer' in f]
    for f in neg_csv_files: 
        computer_features_per_file(os.path.join(neg_path, f), '0')

    dp.T.to_csv('data/anomaly/features.csv', header=None)
    # instead of transpose, do this: http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe


# ? run featurizer per person 
# argparse w/ arguments: input and output directories.



# 1. find the first crossing time:
# t = dp[crossing condition]['time'].min()

# 2. while (dp.length > 0)
#   df[t + 10s] search for crossings every 10 seconds since first crossing
#   










# # 1. read csv
# acc = pd.read_csv('data/theft_classifier_data/theft_CSVs/acc.csv')
# acc['timestamp'] = pd.to_numeric(acc['timestamp'], errors='coerce')
# acc['x'] = pd.to_numeric(acc['x'], errors='coerce')
# acc['y'] = pd.to_numeric(acc['y'], errors='coerce')
# acc['z'] = pd.to_numeric(acc['z'], errors='coerce')
# acc = acc[(np.isfinite(acc['x'])) & (np.isfinite(acc['y'])) & (np.isfinite(acc['z']))]

# # 2. calculate norm sqrt(sum(x^2, y^2, z^2))
# xyz = acc[['x', 'y', 'z']]
# acc['norm'] = np.sqrt(np.square(xyz).sum(axis=1))
# acc['norm_before'] = acc['norm'].shift(1)
# acc['norm_after'] = acc['norm'].shift(-1)

# # 3. find time, t, when value is ~40
# norm_exceed = acc[(acc['norm_before'] < 40) & (acc['norm_after'] > 40) | 
#                   (acc['norm_before'] > 40) & (acc['norm_after'] < 40)]

# # 4. get 1s window before & after t
# time_exceed = norm_exceed['timestamp']
# for t in time_exceed:
#     window_start = t-1000
#     window_end = t+1000
#     # window_start = t-1000 if t-1000 > acc['timestamp'].index[0] else acc['timestamp'].index[0]
#     # window_end = t+1000 if t+1000 < acc['timestamp'].index[acc['timestamp'].size-1] else acc['timestamp'].index[acc['timestamp'].size-1]

#     window1 = acc[(acc['timestamp'] >= window_start) & (acc['timestamp'] <= t)]
#     window1_norms = window1['norm']
#     window2 = acc[(acc['timestamp'] >= t) & (acc['timestamp'] <= window_end)]
#     window2_norms = window2['norm']

#     # print(window_start, t, window_end, acc['timestamp'].index[acc['timestamp'].size-1])

#     # 5. Compute features
#     window1_max, window1_mean, window1_std, window1_rms, window1_arc_len, window1_arc_len_std, window1_mean_abs \
#     = compute_features_per_window(window1, window1_norms)

#     window2_max, window2_mean, window2_std, window2_rms, window2_arc_len, window2_arc_len_std, window2_mean_abs \
#     = compute_features_per_window(window2, window2_norms)

#     # 6. concatenate windows
#     w1 = pd.Series([window1_max, window1_mean, window1_std, window1_rms, window1_arc_len, window1_arc_len_std, window1_mean_abs])
#     w2 = pd.Series([window2_max, window2_mean, window2_std, window2_rms, window2_arc_len, window2_arc_len_std, window2_mean_abs])

#     dp[str(t)+' before'] = w1
#     dp[str(t)+' after'] = w2

# print(dp.T)