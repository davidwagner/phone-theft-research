import csv
import matplotlib.pyplot as plt


featuresfile = 'data/features_win_size_1_2.csv'
# featuresfile = 'data/features.csv'
plot_hist_before = False

plot_tile = featuresfile[5:26] + ('_before' if plot_hist_before else '_after') + '_featues'

pos_max_before = []
pos_mean_before = []
pos_std_before = []
pos_rms_before = []
pos_arc_len_before = []
pos_arc_len_std_before = []
# pos_mean_abs_before = []

pos_max_after = []
pos_mean_after = []
pos_std_after = []
pos_rms_after = []
pos_arc_len_after = []
pos_arc_len_std_after = []
# pos_mean_abs_after = []

neg_max_before = []
neg_max_before = []
neg_mean_before = []
neg_std_before = []
neg_rms_before = []
neg_arc_len_before = []
neg_arc_len_std_before = []
# neg_mean_abs_before = []

neg_max_after = []
neg_max_after = []
neg_mean_after = []
neg_std_after = []
neg_rms_after = []
neg_arc_len_after = []
neg_arc_len_std_after = []
# neg_mean_abs_after = []

with open(featuresfile) as f:
    reader = csv.reader(f)
    for row in reader:
        label = int(row[1])
        if label == 1:
            pos_max_before.append(float(row[2]))
            pos_mean_before.append(float(row[3]))
            pos_std_before.append(float(row[4]))
            pos_rms_before.append(float(row[5]))
            pos_arc_len_before.append(float(row[6]))
            pos_arc_len_std_before.append(float(row[7]))
            # pos_mean_abs_before.append(float(row[8]))

            pos_max_after.append(float(row[8]))
            pos_mean_after.append(float(row[9]))
            pos_std_after.append(float(row[10]))
            pos_rms_after.append(float(row[11]))
            pos_arc_len_after.append(float(row[12]))
            pos_arc_len_std_after.append(float(row[13]))
            # pos_mean_abs_after.append(float(row[15]))

        elif label == 0:
            neg_max_before.append(float(row[2]))
            neg_mean_before.append(float(row[3]))
            neg_std_before.append(float(row[4]))
            neg_rms_before.append(float(row[5]))
            neg_arc_len_before.append(float(row[6]))
            neg_arc_len_std_before.append(float(row[7]))
            # neg_mean_abs_before.append(float(row[8]))

            neg_max_after.append(float(row[8]))
            neg_mean_after.append(float(row[9]))
            neg_std_after.append(float(row[10]))
            neg_rms_after.append(float(row[11]))
            neg_arc_len_after.append(float(row[12]))
            neg_arc_len_std_after.append(float(row[13]))
            # neg_mean_abs_after.append(float(row[15]))

# max_bins = [i for i in range(int(min(min(pos_max),min(neg_max))), int(max(max(pos_max),max(neg_max))) )]
# std_bins = [i for i in range(int(min(min(pos_std),min(neg_std))), int(max(max(pos_std),max(neg_std))) )]
colors = ['red', 'green']
labels = ['pos', 'neg']

if plot_hist_before: 

    fig, axes = plt.subplots(nrows=6, ncols=1)
    plt0, plt1, plt2, plt3, plt4, plt5 = axes.flat

    fig.text(0.5, 0.08, r'$m/s^2$', fontsize=15, ha='center', va='center')
    fig.text(0.105, 0.5, 'percentage of instances', fontsize=15, fontweight='bold', ha='center', va='center', rotation='vertical')

    plt0.hist([pos_max_before, neg_max_before], bins=150, normed=True, color=colors, label=labels)
    plt0.legend(loc='upper center', fontsize=20)
    plt0.set_title('max', fontsize=20, fontweight='bold')
    # plt0.set_xlim(0, 350)
    # plt0.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt0.set_ylabel('percentage of instances', fontsize=15)

    plt1.hist([pos_mean_before, neg_mean_before], bins=150, normed=True, color=colors, label=labels)
    plt1.legend(loc='upper center', fontsize=20)
    plt1.set_title('mean', fontsize=20, fontweight='bold')
    # plt1.set_xlim(0, 70)
    # plt1.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt1.set_ylabel('percentage of instances', fontsize=15)

    plt2.hist([pos_std_before, neg_std_before], bins=150, normed=True, color=colors, label=labels)
    plt2.legend(loc='upper center', fontsize=20)
    plt2.set_title('std', fontsize=20, fontweight='bold')
    # plt2.set_xlim(0, 45)
    # plt2.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt2.set_ylabel('percentage of instances', fontsize=15)

    plt3.hist([pos_rms_before, neg_rms_before], bins=150, normed=True, color=colors, label=labels)
    plt3.legend(loc='upper center', fontsize=20)
    plt3.set_title('rms', fontsize=20, fontweight='bold')
    # plt3.set_xlim(0, 80)
    # plt3.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt3.set_ylabel('percentage of instances', fontsize=15)

    plt4.hist([pos_arc_len_before, neg_arc_len_before], bins=150, normed=True, color=colors, label=labels)
    plt4.legend(loc='upper center', fontsize=20)
    plt4.set_title('arc_len', fontsize=20, fontweight='bold')
    # plt4.set_xlim(0, 23)
    # plt4.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt4.set_ylabel('percentage of instances', fontsize=15)

    plt5.hist([pos_arc_len_std_before, neg_arc_len_std_before], bins=150, normed=True, color=colors, label=labels)
    plt5.legend(loc='upper center', fontsize=20)
    plt5.set_title('arc_len_std', fontsize=20, fontweight='bold')
    # plt5.set_xlim(0, 800)
    # plt5.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt5.set_ylabel('percentage of instances', fontsize=15)

    # plt6.hist([pos_mean_abs_before, neg_mean_abs_before], bins=150, normed=True, color=colors, label=labels)
    # plt6.legend(loc='upper right')
    # plt6.set_title('mean_abs')

    # fig.suptitle(plot_tile)
    plt.show()

if not plot_hist_before: 

    fig, axes = plt.subplots(nrows=6, ncols=1)
    plt0, plt1, plt2, plt3, plt4, plt5 = axes.flat

    fig.text(0.5, 0.08, r'$m/s^2$', fontsize=15, ha='center', va='center')
    fig.text(0.105, 0.5, 'percentage of instances', fontsize=15, fontweight='bold', ha='center', va='center', rotation='vertical')

    plt0.hist([pos_max_after, neg_max_after], bins=150, normed=True, color=colors, label=labels) # bins=150, 
    plt0.legend(loc='upper center', fontsize=20)
    plt0.set_title('max', fontsize=20, fontweight='bold')
    # plt5.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt5.set_ylabel('percentage of instances', fontsize=15)

    plt1.hist([pos_mean_after, neg_mean_after], bins=150, normed=True, color=colors, label=labels)
    plt1.legend(loc='upper center', fontsize=20)
    plt1.set_title('mean', fontsize=20, fontweight='bold')
    # plt5.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt5.set_ylabel('percentage of instances', fontsize=15)

    plt2.hist([pos_std_after, neg_std_after], bins=150, normed=True, color=colors, label=labels)
    plt2.legend(loc='upper center', fontsize=20)
    plt2.set_title('std', fontsize=20, fontweight='bold')
    # plt5.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt5.set_ylabel('percentage of instances', fontsize=15)

    plt3.hist([pos_rms_after, neg_rms_after], bins=150, normed=True, color=colors, label=labels)
    plt3.legend(loc='upper center', fontsize=20)
    plt3.set_title('rms', fontsize=20, fontweight='bold')
    # plt5.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt5.set_ylabel('percentage of instances', fontsize=15)

    plt4.hist([pos_arc_len_after, neg_arc_len_after], bins=150, normed=True, color=colors, label=labels)
    plt4.legend(loc='upper center', fontsize=20)
    plt4.set_title('arc_len', fontsize=20, fontweight='bold')
    # plt5.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt5.set_ylabel('percentage of instances', fontsize=15)

    plt5.hist([pos_arc_len_std_after, neg_arc_len_std_after], bins=150, normed=True, color=colors, label=labels)
    plt5.legend(loc='upper center', fontsize=20)
    plt5.set_title('arc_len_std', fontsize=20, fontweight='bold')
    # plt5.set_xlabel(r'$m/s^2$', fontsize=15)
    # plt5.set_ylabel('percentage of instances', fontsize=15)

    # plt6.hist([pos_mean_abs_after, neg_mean_abs_after], bins=150, normed=True, color=colors, label=labels)
    # plt6.legend(loc='upper right')
    # plt6.set_title('mean_abs')

    # fig.suptitle(plot_tile)
    plt.show()

# plt.hist(pos_max, bins=100, normed=True)
# plt.show()
# plt.hist(pos_std, bins=100, normed=True)
# plt.show()
# plt.hist(neg_max, bins=100, normed=True)
# plt.xticks([0, 10, 20, 40, 50, 60, 70, 80, 90, 100])
# plt.show()
# plt.hist(neg_std, bins=100, normed=True)
# plt.show()