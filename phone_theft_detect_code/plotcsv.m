close all;
close all hidden;
path = '/Users/JasonLiu/research/security/phone_theft_detect/data/theft_classifier_data/';
% filename = 'theft_CSVs/AppMon_81452402_BatchedAccelerometer_2016_07_17_23_13_12_.csv';
% filename = 'theft_CSVs/AppMon_e5b921a6-d94c-43a9-a38a-129bfebcdac9_BatchedAccelerometer_2016_12_04_19_24_47_.csv';
filename = 'neg_CSVS/AppMon_947c57eb-d681-455c-aa92-8397c103c102_BatchedAccelerometer_2016_12_04_09_27_34_.csv';
% filename = 'neg_CSVS/AppMon_72308974-27d4-452c-8ddb-752d5ea15c31_BatchedAccelerometer_2016_12_05_12_16_19_.csv';
f = strcat(path,filename);
data = csvread(f);
t = data(:,1);
x = data(:,2);
y = data(:,3);
z = data(:,4);
A = [x y z];
n = sqrt(sum(A.^2,2)); % dot op '.' : element wise op, ^2.

% shade 1s windows before & after first time when acc norm exceeds 40
% % https://www.mathworks.com/help/matlab/ref/area.html
% % https://www.mathworks.com/help/matlab/creating_plots/compare-data-sets-using-overlayed-area-graphs-1.html
% t_before = 968931:969931;
% t_after = 969931:970931;
% acc_pos = ones(length(t_before), 1) * 120;
% acc_neg = ones(length(t_before), 1) * -80;
% area(t_before,acc_pos,'FaceColor',[1 1 0.2],'EdgeColor',[1 1 0.2]);
% hold on;
% area(t_before,acc_neg,'FaceColor',[1 1 0.2],'EdgeColor',[1 1 0.2]);
% hold on;
% area(t_after,acc_pos,'FaceColor',[0.4 1 0.4],'EdgeColor',[0.4 1 0.4]);
% hold on;
% area(t_after,acc_neg,'FaceColor',[0.4 1 0.4],'EdgeColor',[0.4 1 0.4]);

hold on;
yticks([-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]);
h = plot(t,x,'r', t,y,'b', t,z,'g', t,n,'m');
hold off;
title('accelerometer data (x, y, z, norm)');
xlabel('ms');
ylabel('m/s^2');
legend('X','Y','Z','Norm');

% when view plot: Tools -> Options -> Horizontal Zoom




