close all
close all hidden
path = '/Users/JasonLiu/research/security/phone_theft_detect/data/theft_classifier_data/';
% filename = 'theft_CSVs/AppMon_81452402_BatchedAccelerometer_2016_07_17_23_13_12_.csv'; % used to plot pos acc in the paper
% filename = 'theft_CSVs/AppMon_e5b921a6-d94c-43a9-a38a-129bfebcdac9_BatchedAccelerometer_2016_12_04_19_24_47_.csv';

% filename = 'neg_CSVS/AppMon_72308974-27d4-452c-8ddb-752d5ea15c31_BatchedAccelerometer_2016_12_05_12_16_19_.csv';
% filename = 'neg_CSVS/AppMon_53d9901d-c6fe-4ffa-9af5-5136d2443595_BatchedAccelerometer_2016_10_06_11_06_41_.csv';
filename = 'neg_CSVS/AppMon_241eeff5-125c-493d-8217-eafedd17855c_BatchedAccelerometer_2016_10_27_21_41_31_.csv'; % used to plot neg acc in the paper

f = strcat(path,filename);
data = csvread(f);
% t = data(:,1);
% x = data(:,2);
% y = data(:,3);
% z = data(:,4);
% A = [x y z];
% n = sqrt(sum(A.^2,2)); % dot op '.' : element wise op, ^2.




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






% hold on
% yticks([-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
% h = plot(t,x,'r', t,y,'b', t,z,'g', t,n,'m');
% hold off
% title('accelerometer data (x, y, z, norm)')
% xlabel('ms')
% ylabel('m/s^2')
% legend('X','Y','Z','Norm')

% when view plot: Tools -> Options -> Horizontal Zoom







% plot x, y, z and L2 norm of acceleration of one theft instance separately
% theft instance: 
% t = data(116616:122616,1);
% x = data(116616:122616,2);
% y = data(116616:122616,3);
% z = data(116616:122616,4);
% negative instance:
t = data(1:356036,1);
x = data(1:356036,2);
y = data(1:356036,3);
z = data(1:356036,4);
A = [x y z];
n = sqrt(sum(A.^2,2));

figure
x_plot = subplot(1,4,1);
plot(t,x,'r')
title('x acceleration')
xlabel('ms')
ylabel('m/s^2')

y_plot = subplot(1,4,2);
plot(t,y,'b')
title('y acceleration')
xlabel('ms')
ylabel('m/s^2')

z_plot = subplot(1,4,3);
plot(t,z,'g')
title('z acceleration')
xlabel('ms')
ylabel('m/s^2')

n_plot = subplot(1,4,4);
plot(t,n,'m')
title('magnitude of acceleration')
xlabel('ms')
ylabel('m/s^2')

linkaxes([x_plot,y_plot,z_plot,n_plot], 'xy')
ylim([-55 165])

