"""
Generate dataset for classifiers.
Run on theserver: ssh jasonxyliu@theserver.cs.berkeley.edu
Author: Jason Liu
"""

import pandas as pd
import numpy as np
import os

ACC_THRESHOLD = 40 # trigger condition.
WIN_SIZE_BEFORE = 1 # number of seconds before first 40-speak.
WIN_SIZE_AFTER = 2 # number of seconds after first 40-speak.
WIN_LEN = 1000 # fixed sample frequency of accelerometer.

MUL_AFTER_WINS = False # whether to use Xs after-window or Y # of consecutive 1s after-windows
NUM_AFTER_WINS = 1 # number of consecutive after-windows

def compute_features_per_window(window, window_norms):
	w_max = window_norms.max()
	w_mean = window_norms.mean()
	w_std = window_norms.std()
	w_rms = np.sqrt(np.square(window_norms).mean())
    # w_rms = np.sqrt(np.square(window_norms).sum())

	w_nonans = window[np.isfinite(window['norm_after'])]
	w_arc_len = (w_nonans['norm_after'] - w_nonans['norm']).abs().sum() / w_nonans['norm'].size

	w_arc_len_std = w_arc_len * w_std
	# w_mean_abs = window_norms.abs().mean()

	return w_max, w_mean, w_std, w_rms, w_arc_len, w_arc_len_std

def computer_features_per_file(file_name, class_label):
    print(file_name)

    # 1. read csv
    acc = pd.read_csv(file_name, names=['timestamp', 'x', 'y', 'z'], dtype={'timestamp':np.float64, 'x':np.float64, 'y':np.float64, 'z':np.float64}, index_col=False)
    # print(acc.head(5))

    acc = acc[(pd.notnull(acc['x'])) & (pd.notnull(acc['z'])) & (pd.notnull(acc['z']))]
    # acc = acc[(np.isfinite(acc['x'])) & (np.isfinite(acc['y'])) & (np.isfinite(acc['z']))]

    # 2. calculate norm, sqrt(sum(x^2, y^2, z^2))
    xyz = acc[['x', 'y', 'z']]
    acc['norm'] = np.sqrt(np.square(xyz).sum(axis=1))
    acc['norm_before'] = acc['norm'].shift(1)
    acc['norm_after'] = acc['norm'].shift(-1)

    # 3. find all times, t, when value is ~40, i.e. find all upward crossings.
    norm_exceed = acc[(acc['norm_before'] < ACC_THRESHOLD) & (acc['norm'] >= ACC_THRESHOLD)]

    # 4. get 1s window before & after t
    time_crossing = norm_exceed['timestamp']

    # due to pos experiment noise, we crop times norm cross 40 manully
    if class_label == '1':
        if '81452402' in file_name:
            time_exceed = np.array([161330.0, 216094.0, 287314.0, 353417.0, 416721.0, 475336.0, 535971.0, 596367.0, 656234.0, 726406.0,
                                    785748.0, 848996.0, 909128.0, 969941.0, 1028077.0, 1092756.0, 1148382.0, 1211740.0, 1269078.0, 1329543.0,
                                    1967731.0, 2048085.0, 2136125.0, 2221992.0, 2298554.0, 2372050.0, 2444078.0, 2514334.0, 2592409.0, 2671525.0])
        elif '2016_12_04_19_24_47' in file_name:
            time_exceed = np.array([933876.0, 966708.0, 1013558.0, 1061703.0, 1101594.0, 1151936.0, 1195055.0, 1247129.0, 1292453.0, 1361568.0,
                                    1432105.0, 1500765.0, 1544397.0, 1595380.0, 1637371.0, 1697678.0, 1758845.0, 1822171.0, 1893906.0, 1951704.0,
                                    2018429.0, 2088872.0, 2131228.0, 2172948.0, 2228841.0, 2284761.0, 2342263.0, 2389745.0, 2439722.0, 2483499.0])
                                  # 2018429.0, 2088872.0, 2128050.0, 2172948.0, 2223263.0, 2284761.0, 2342263.0, 2389745.0, 2439722.0, 2483499.0])
        elif '2017_10_06_17_30_03' in file_name:
            time_exceed = np.array([5146544.0, 5183833.0, 5218368.0, 5257953.0, 5300411.0, 5343136.0, 5382330.0, 5439306.0, 5477007.0, 5513164.0,
                                    5545777.0, 5577401.0, 5607503.0, 5643477.0, 5685244.0, 5713794.0, 5740740.0, 5785859.0, 5824875.0, 5863780.0,
                                    5897162.0, 5929834.0, 5981464.0, 6024385.0, 6064576.0, 6112529.0, 6171044.0, 6204153.0, 6243535.0, 6286688.0])
        elif '2017_10_06_20_06_48' in file_name:
            time_exceed = np.array([6667210.0, 6716574.0, 6777556.0, 6827141.0, 6877395.0, 6930686.0, 6991879.0, 7049268.0, 7095749.0, 7148348.0,
                                    7216223.0, 7285226.0, 7360614.0, 7436579.0, 7500386.0, 7575378.0, 7637309.0, 7703844.0, 7766888.0, 7827128.0])
        elif '2017_10_07_15_28_57' in file_name:
            time_exceed = np.array([511982.0, 560226.0, 611987.0, 651113.0, 693133.0, 744144.0, 800825.0, 855893.0, 926166.0, 1006527.0])
        elif '2017_12_04_20_45_34' in file_name:
            print('Nexus 6P Exp 2.2')
            time_exceed = np.array([6899882.0, 6975954.0, 7044713.0, 7110112.0, 7184112.0,
                                    7249911.0, 7313394.0, 7372449.0, 7435753.0, 7504721.0,
                                    7553743.0, 7624700.0, 7689427.0, 7757450.0, 7812858.0])
        elif '2017_12_04_21_52_33' in file_name:
            print('Nexus 6P Exp 2.3&2.1')
            time_exceed = np.array([10941787.0, 10866409.0, 11045347.0, 11136426.0, 11211593.0,
                                    11315076.0, 11393353.0, 11491042.0, 11626581.0, 11721637.0,
                                    11802508.0, 11934919.0, 12007584.0, 12095546.0, 12158412.0,
                                    12831481.0, 12879822.0, 12945946.0, 13010587.0, 13068502.0,
                                    13234685.0, 13269032.0, 13398456.0, 13450762.0, 13520260.0,
                                    13569496.0, 13632456.0, 13691345.0, 13730728.0, 13781298.0])
        else:
            print('unexpected pos file')
        # print(pd.Series(time_exceed).isin(time_crossing.tolist()))
    else:
        time_exceed = time_crossing

    # old featurizer: different for pos and neg datasets.
    for t in time_exceed:
        DP_IMCOMPLETE = False

        # 5.1 generate features from 1s window before 40-spike
        window0_start = t - WIN_SIZE_BEFORE * WIN_LEN
        window0 = acc[(acc['timestamp'] >= window0_start) & (acc['timestamp'] < t)]
        window0_norms = window0['norm']
        if window0.shape[0] > 1:
            # compute features
            window0_max, window0_mean, window0_std, window0_rms, window0_arc_len, window0_arc_len_std \
            = compute_features_per_window(window0, window0_norms)
            # construct partial date point from features
            w0 = pd.Series([window0_max, window0_mean, window0_std, window0_rms, window0_arc_len, window0_arc_len_std])

            # 6.1 concatenate to construct a data point
            label = pd.Series([class_label])
            data_point = pd.concat([label, w0])

            # 5.2 generate features from each 1s window after 40-spike (# of 1s window after 40-spike >=1).
            window_after_start = t
            for _ in range(NUM_AFTER_WINS):
                window_after_end = window_after_start + WIN_SIZE_AFTER * WIN_LEN

                window = acc[(acc['timestamp'] >= window_after_start) & (acc['timestamp'] < window_after_end)]
                window_norms = window['norm']
                if window.shape[0] > 1:
                    # compute features
                    window_max, window_mean, window_std, window_rms, window_arc_len, window_arc_len_std \
                    = compute_features_per_window(window, window_norms)
                    # construct partial date point from features
                    w = pd.Series([window_max, window_mean, window_std, window_rms, window_arc_len, window_arc_len_std])

                    window_after_start = window_after_end

                    # 6.2 Concatenate to construct a data point
                    data_point = pd.concat([data_point, w])
                else:
                    window_after_start = window_after_end
                    DP_IMCOMPLETE = True
                    break

            if not DP_IMCOMPLETE:
                dp[file_name+'_'+str(t)] = data_point



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

    # # neg_path = 'data/theft_classifier_data/neg_CSVs' # local
    # neg_path = 'data/theft_classifier_data/sensor_research' # remote
    # neg_csv_files = [f for f in os.listdir(neg_path) if f.endswith('.csv') and 'BatchedAccelerometer' in f]
    # for f in neg_csv_files:
    #     computer_features_per_file(os.path.join(neg_path, f), '0')

    if MUL_AFTER_WINS:
        dp.T.to_csv('data/features_win_size_{}_({}){}.csv'.format(WIN_SIZE_BEFORE, NUM_AFTER_WINS, WIN_SIZE_AFTER), header=True)
    else:
        print(dp.T.shape)
        dp.T.to_csv('data/features_win_size_{}_{}_new_pos.csv'.format(WIN_SIZE_BEFORE, WIN_SIZE_AFTER), header=True)
        # instead of transpose, do this: http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe


# ? run featurizer per person
# argparse w/ arguments: input and output directories.



# 1. find the first crossing time:
# t = dp[crossing condition]['time'].min()

# 2. while (dp.length > 0)
#   df[t + 10s] search for crossings every 10 seconds since first crossing
