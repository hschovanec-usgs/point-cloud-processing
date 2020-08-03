clear all;
close all;

first_date = input('What is the first date? (In the format yyyyMMdd)', 's') ;
second_date = input('What is the second date? (In the format yyyyMMdd)', 's') ;
day_one = datetime(first_date,'InputFormat','yyyyMMdd');
day_two = datetime(second_date,'InputFormat','yyyyMMdd');
num_days = days(day_two-day_one);
d1 = first_date;
d2 = second_date;
filepattern = strcat( first_date, '-', second_date, '\\falls\\fall__*')
files = dir(filepattern);

premier_nuage = 1;
dernier_nuage = length(files);

ll = dernier_nuage - premier_nuage + 1;
vnum = [];
numPoints = [];
vols = [];
densities = [];
lx = [];
ly = [];
lz = [];
avg_change = [];
sd_change = [];
min_change = [];
max_change = [];
rockfall = [];
centx = [];
centy = [];
centz = [];

for j=premier_nuage:1:dernier_nuage
    fall_number_herer = j
   close all
   clear M
   clear N
   clear A
   clear V
    
   clc
    
if j<10
    M=dlmread(strcat( first_date, '-', second_date, '\\falls\\fall__00000',num2str(j),'.txt'));
elseif j<100
    M=dlmread(strcat(first_date, '-', second_date, '\\falls\\fall__0000',num2str(j),'.txt'));

elseif j <1000
    M=dlmread(strcat( first_date, '-', second_date, '\\falls\\fall__000',num2str(j),'.txt'));
else
     M=dlmread(strcat( first_date, '-', second_date, '\\falls\\fall__00',num2str(j),'.txt'));
end

disp(strcat(first_date, '-', second_date, 'fall ',num2str(j)));

N=M(:,1:3);
 
 
 %% --------------------------STATS--------------------------
 vnum(j, 1) = j;
 date1(j, 1) = string(d1);
 date2(j, 1) = string(d2);
 rockfall(j, 1) = 1;
 cx = mean(N(:, 1));
 cy = mean(N(:, 2));
 cz = mean(N(:, 3));
 centx(j,1) = cx;
 centy(j,1) = cy;
 centz(j,1) = cz;
 change = M(:,4);
 avg_change(j, 1) = nanmean(abs(change));
 sd_change(j, 1) = nanstd(change);
 min_change(j, 1) = nanmin(change);
 max_change(j, 1) = nanmax(change);
 c = [cx cy cz];
 numPoints(j, 1) = length(N);
 cfs = pca(N);
 cfa = [cfs(1,1) cfs(2,1) cfs(3,1)];
 cfb = [cfs(1,2) cfs(2,2) cfs(3,2)];
 cfc = [cfs(1,3) cfs(2,3) cfs(3,3)];
 maga = sqrt(cfs(1,1)^2 + cfs(2,1)^2 + cfs(3,1)^2);
 magb = sqrt(cfs(1,2)^2 + cfs(2,2)^2 + cfs(3,2)^2); 
 magc = sqrt(cfs(1,3)^2 + cfs(2,3)^2 + cfs(3,3)^2);
 dap = [];
 dan = [];
 dbp = [];
 dbn = [];
 dcp = [];
 dcn = [];
 for i=1:length(N)
    x = N(i, 1);
    y = N(i, 2);
    z = N(i, 3);
    v = [x-cx y-cy z-cz];
    dad = dot(v, cfa);
    dbd = dot(v, cfb);
    dcd = dot(v, cfc);
    if dad > 0
        dap(i) = dad;
    else
        dan(i) = dad;
    end
    if dbd > 0
        dbp(i) = dbd;
    else
        dbn(i) = dbd;
    end
    if dcd > 0
        dcp(i) = dcd;
    else
        dcn(i) = dcd;
    end
        
 end
 da = nanmax(dap) + nanmax(abs(dan));
 db = nanmax(dbp) + nanmax(abs(dbn));
 dc = nanmax(dcp) + nanmax(abs(dcn));
 
lx(j, 1) = da;
ly(j, 1) = db;
lz(j, 1) = dc;

figure('visible','off')
V = [];
r=0:0.02:3;
for i=1:length(r)
    radius_loop = r(i);
    v=alphavol_true_modif(N,radius_loop,1);
    A(i)=radius_loop;
    V(i)=v ;
    i=i+1;
end



if max(V) < 0.002
    pt = prctile(V, 10);
elseif max(V) < 0.02
    pt = prctile(V, 20);
elseif max(V) < 0.2
    pt = prctile(V, 40);
else
    pt = prctile(V, 50);
end
[minValue,idx] = min(abs(pt - V));
ap = A(idx);
vp = V(idx);
vols(j, 1) = vp;

densities(j,1) = vp/length(N);
end

%% -----------------------------------------------------------------------
%% FILTERING

vol_file = importdata("volumesUpdated.xlsx");
vol_sheet = vol_file.data.Sheet1;
num_test = vol_sheet(:, 1);
num_pts_test = vol_sheet(:, 2);
volume_test = vol_sheet(:, 3);
density_test = vol_sheet(:, 4);
lx_test = vol_sheet(:, 5);
ly_test = vol_sheet(:, 6);
lz_test = vol_sheet(:, 7);
avg_abs_change_test = vol_sheet(:, 8);
SD_change_test = vol_sheet(:, 9);
max_change_test = vol_sheet(:, 10);
min_change_test = vol_sheet(:, 11);
rock_fall_test = vol_sheet(:, 14);

idx_v = find(volume_test > 0);
vv = volume_test(idx_v);
np = num_pts_test(idx_v);
dd = density_test(idx_v);
av = avg_abs_change_test(idx_v);
sd = SD_change_test(idx_v);
rf = rock_fall_test(idx_v);

% Setting up training and testing data --------------------------------
TBL = zeros(length(rf), 5);
TBL(:,1) = np;
TBL(:,2) = vv;
TBL(:,3) = dd;
TBL(:,4) = av;
TBL(:,5) = sd;

data_train = TBL(:,:);
rock_fall_train = rf(:,:);
% -----------------------------------------------------------------
% Random Forest model
tree_mdl = TreeBagger(100, data_train, ...
rock_fall_train, 'PredictorSelection', 'interaction-curvature', ...
'OOBPredictorImportance', 'on', 'OOBPrediction', 'on', ...
'MaxNumSplits', 100);
% 
% 
%% Classify
% % Setting up training and testing data --------------------------------
TBL = zeros(length(numPoints), 5);
TBL(:,1) = numPoints';
TBL(:,2) = vols;
TBL(:,3) = densities;
TBL(:,4) = avg_change;
TBL(:,5) = sd_change;

[tree_label, tree_score] = predict(tree_mdl, TBL);
tree_probability = tree_score(:,2);
tree_thresholded_score = tree_probability > 0.325;


filename = strcat( first_date, '-', second_date, '\\volumes.xlsx');
table_array = [date1, date2, vnum, numPoints, vols, densities, centx, centy, ...
    centz, lx, ly, lz, avg_change, sd_change, max_change, min_change, tree_thresholded_score];
T = array2table(table_array,...
    'VariableNames',{'Date1', 'Date2', 'Volume_Number', 'Number_of_Points','Volume','Density',...
    'Cx', 'Cy', 'Cz', 'Lx', 'Ly', 'Lz', 'Average_Absolute_Change', ...
    'SD_Change', 'Maximum_Change', 'Minimum_Change', 'Rockfall'});
writetable(T,filename,'Sheet',1)


vol_fig = vols;
npts = numPoints;
rf = tree_thresholded_score;
idxt = find(rf == 1);
ft = vol_fig(idxt);
idxf = find(rf == 0);
ff = vols(idxf);
npf = npts(idxf);


figure(1)
subplot(2, 2, 1)
step = 365.25/num_days;
msteps = 1:1:length(ft);
ysteps = msteps * step;
loglog(sort(ft, 'descend'), ysteps, '.');
ylabel("Cumulative Number of Events Per Year");
xlabel("Rockfall Volume (cubic meter)");
title('Filtered');


subplot(2, 2, 2)
step = 30.4167/num_days;
msteps = 1:1:length(ft);
msteps = msteps * step;
loglog(sort(ft, 'descend'), msteps, '.');
ylabel("Cumulative Number of Events Per Month");
xlabel("Rockfall Volume (cubic meter)");
title('Filtered');

ff = vols;

subplot(2, 2, 3)
step = 365.25/num_days;
ysteps = 1:1:length(ff);
ysteps = ysteps * step;
loglog(sort(ff, 'descend'), ysteps, '.');
ylabel("Cumulative Number of Events Per Year");
xlabel("Rockfall Volume (cubic meter)");
title('All Volumes');

subplot(2, 2, 4)
step = 30.4167/num_days;
msteps = 1:1:length(ff);
msteps = msteps * step;  
loglog(sort(ff, 'descend'), msteps, '.');
ylabel("Cumulative Number of Events Per Month");
xlabel("Rockfall Volume (cubic meter)");
title('All Volumes');

savefig('cumulative.fig')
