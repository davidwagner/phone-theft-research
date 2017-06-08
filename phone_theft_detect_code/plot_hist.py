import csv
import matplotlib.pyplot as plt


featuresfile = 'data/features_win_size_1_1.csv'
# featuresfile = 'data/features.csv'

pos_max_before = []
pos_mean_before = []
pos_std_before = []
pos_rms_before = []
pos_arc_len_before = []
pos_arc_len_std_before = []
pos_mean_abs_before = []

pos_max_after = []
pos_mean_after = []
pos_std_after = []
pos_rms_after = []
pos_arc_len_after = []
pos_arc_len_std_after = []
pos_mean_abs_after = []

neg_max_before = []
neg_max_before = []
neg_mean_before = []
neg_std_before = []
neg_rms_before = []
neg_arc_len_before = []
neg_arc_len_std_before = []
neg_mean_abs_before = []

neg_max_after = []
neg_max_after = []
neg_mean_after = []
neg_std_after = []
neg_rms_after = []
neg_arc_len_after = []
neg_arc_len_std_after = []
neg_mean_abs_after = []

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
            pos_mean_abs_before.append(float(row[8]))

            pos_max_after.append(float(row[9]))
            pos_mean_after.append(float(row[10]))
            pos_std_after.append(float(row[11]))
            pos_rms_after.append(float(row[12]))
            pos_arc_len_after.append(float(row[13]))
            pos_arc_len_std_after.append(float(row[14]))
            pos_mean_abs_after.append(float(row[15]))

        elif label == 0:
            neg_max_before.append(float(row[2]))
            neg_mean_before.append(float(row[3]))
            neg_std_before.append(float(row[4]))
            neg_rms_before.append(float(row[5]))
            neg_arc_len_before.append(float(row[6]))
            neg_arc_len_std_before.append(float(row[7]))
            neg_mean_abs_before.append(float(row[8]))

            neg_max_after.append(float(row[9]))
            neg_mean_after.append(float(row[10]))
            neg_std_after.append(float(row[11]))
            neg_rms_after.append(float(row[12]))
            neg_arc_len_after.append(float(row[13]))
            neg_arc_len_std_after.append(float(row[14]))
            neg_mean_abs_after.append(float(row[15]))

# max_bins = [i for i in range(int(min(min(pos_max),min(neg_max))), int(max(max(pos_max),max(neg_max))) )]
# std_bins = [i for i in range(int(min(min(pos_std),min(neg_std))), int(max(max(pos_std),max(neg_std))) )]
colors = ['red', 'green']
labels = ['pos', 'neg']

fig, axes = plt.subplots(nrows=7, ncols=1)
plt0, plt1, plt2, plt3, plt4, plt5, plt6 = axes.flat

plt0.hist([pos_max_before, neg_max_before], bins=150, normed=True, color=colors, label=labels)
plt0.legend(loc='upper right')
plt0.set_title('max')

plt1.hist([pos_mean_before, neg_mean_before], bins=150, normed=True, color=colors, label=labels)
plt1.legend(loc='upper right')
plt1.set_title('mean')

plt2.hist([pos_std_before, neg_std_before], bins=150, normed=True, color=colors, label=labels)
plt2.legend(loc='upper right')
plt2.set_title('std')

plt3.hist([pos_rms_before, neg_rms_before], bins=150, normed=True, color=colors, label=labels)
plt3.legend(loc='upper right')
plt3.set_title('rms')

plt4.hist([pos_arc_len_before, neg_arc_len_before], bins=150, normed=True, color=colors, label=labels)
plt4.legend(loc='upper right')
plt4.set_title('arc_len')

plt5.hist([pos_arc_len_std_before, neg_arc_len_std_before], bins=150, normed=True, color=colors, label=labels)
plt5.legend(loc='upper right')
plt5.set_title('arc_len_std')

plt6.hist([pos_mean_abs_before, neg_mean_abs_before], bins=150, normed=True, color=colors, label=labels)
plt6.legend(loc='upper right')
plt6.set_title('mean_abs')

fig.suptitle("features")
plt.show()

# fig, axes = plt.subplots(nrows=7, ncols=1)
# plt0, plt1, plt2, plt3, plt4, plt5, plt6 = axes.flat

# plt0.hist([pos_max_after, neg_max_after], bins=150, normed=True, color=colors, label=labels)
# plt0.legend(loc='upper right')
# plt0.set_title('max')

# plt1.hist([pos_mean_after, neg_mean_after], bins=150, normed=True, color=colors, label=labels)
# plt1.legend(loc='upper right')
# plt1.set_title('mean')

# plt2.hist([pos_std_after, neg_std_after], bins=150, normed=True, color=colors, label=labels)
# plt2.legend(loc='upper right')
# plt2.set_title('std')

# plt3.hist([pos_rms_after, neg_rms_after], bins=150, normed=True, color=colors, label=labels)
# plt3.legend(loc='upper right')
# plt3.set_title('rms')

# plt4.hist([pos_arc_len_after, neg_arc_len_after], bins=150, normed=True, color=colors, label=labels)
# plt4.legend(loc='upper right')
# plt4.set_title('arc_len')

# plt5.hist([pos_arc_len_std_after, neg_arc_len_std_after], bins=150, normed=True, color=colors, label=labels)
# plt5.legend(loc='upper right')
# plt5.set_title('arc_len_std')

# plt6.hist([pos_mean_abs_after, neg_mean_abs_after], bins=150, normed=True, color=colors, label=labels)
# plt6.legend(loc='upper right')
# plt6.set_title('mean_abs')

# fig.suptitle("features")
# plt.show()

# plt.hist(pos_max, bins=100, normed=True)
# plt.show()
# plt.hist(pos_std, bins=100, normed=True)
# plt.show()
# plt.hist(neg_max, bins=100, normed=True)
# plt.xticks([0, 10, 20, 40, 50, 60, 70, 80, 90, 100])
# plt.show()
# plt.hist(neg_std, bins=100, normed=True)
# plt.show()