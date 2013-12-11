% Script to run ICDM experiments with sentiment data
clear; clc; close all;

%% General setup
% Read in data
load('../data/icdm-20newsgroup.mat')

Xt = Xk(splitk~=3,:);
yt = yk(splitk~=3);

Src{1}.X = Xk(splitk==3,:);
Src{1}.y = yk(splitk==3);
Src{2}.X = Xe(splite==3,:);
Src{2}.y = ye(splite==3);
Src{3}.X = Xd(splitd==3,:);
Src{3}.y = yd(splitd==3);
alphas = [ 0.4 0.6 0.8 ]

kfolds = 5;
reps = 5;
parts = data_kfold(yt, 10, 1);
T = tabulate(parts);
budget = 0;
for s=1:kfolds
    budget = max(budget, sum(parts~=s));
end

base_learner = @svm_linearl2l2;
c0_iwal = 128;
c1 = 1; %5 + 2*sqrt(2);
c2 = 1; %5;
c0_tiwal = 128;

sl_err  = zeros(1, kfolds);
tlu_err = zeros(numel(Src), kfolds);
tls_err = zeros(numel(Src), kfolds);
dadist = zeros(numel(Src), 1);
crisk = zeros(numel(Src), 1);

iwal_query_cts  = zeros(budget+1, kfolds*reps);
iwal_err_seq    = zeros(budget+1, kfolds*reps);
tiwal_query_cts = zeros(numel(Src), budget+1, kfolds*reps);
tiwal_err_seq   = zeros(numel(Src), budget+1, kfolds*reps);

for s=1:kfolds
    fprintf('SPLIT %3d\n', s)

    Xtr = Xt(parts~=s,:);
    ytr = yt(parts~=s);
    Xte = Xt(parts==s,:);
    yte = yt(parts==s);
    n = length(ytr);
    
    %% test error, fully supervised
    [ h, ~ ] = base_learner(Xtr, ytr, ones(size(ytr)), struct());
    [ yhat, ~ ] = h(Xte);
    sl_err(s) = err(yte, yhat);

    for r=1:reps
        fprintf('REP %3d\n', r)

        %% shuffle training data
        query_idx = randperm(n);
        one_of_each = [ find(ytr(query_idx)>0, 1, 'first') find(ytr(query_idx)<0, 1, 'first') ];
        query_idx = [ query_idx(one_of_each) query_idx(setdiff(1:n, one_of_each)) ];
        [ ~, sort_idx ] = sort(query_idx);

        Xq = Xtr(query_idx,:);
        yq = ytr(query_idx);
        
        %% run IWAL CAL
        fprintf('Run IWAL CAL...\n')
        [ y_al, iw_al, hist_al ] = iwal_cal(Xq, nan(size(yq)), yq, budget, ...
                                            base_learner, c0_iwal, c1, c2, ...
                                            Xte, yte, struct('quiet', 1));
        fprintf('DONE!\n')

        % record performance
        iwal_query_cts(:,(s-1) * reps + r) = cumsum(hist_al(:,1),1);
        iwal_err_seq(:,(s-1) * reps + r) = hist_al(:,5);

        for src=1:numel(Src)
            fprintf('SRC %3d\n', src)
            
            Xs = Src{src}.X;
            ys = Src{src}.y;
            m = length(ys);
            alp = alphas(src);

            if r==1
                dadist(src) = approx_da_distance(Xs, Xtr);
                crisk(src) = approx_combined_risk(Xs, ys, Xtr, ytr);

                %% test error, unsupervised transfer
                ws = tiwal_cal_weights(ones(size(ys)), m, struct('m', m, 'alpha', alp));
                [ h, ~ ] = base_learner(Xs, ys, ws, struct());
                [ yhat, ~ ] = h(Xte);
                tlu_err(src, s) = err(yte, yhat);

                %% test error, fully supervised transfer
                X = [ Xs; Xtr ];
                y = [ ys; ytr ];
                w = tiwal_cal_weights(ones(size(y)), m + n, struct('m', m, 'alpha', alp));
                [ h, ~ ] = base_learner(X, y, w, struct());
                [ yhat, ~ ] = h(Xte);
                tls_err(src, s) = err(yte, yhat);
            end

            %% run TIWAL CAL
            fprintf('Run TIWAL CAL...\n')
            [ y_tal, iw_tal, hist_tal ] = tiwal_cal(Xq, nan(size(yq)), yq, ...
                                                    budget, base_learner, ...
                                                    c0_tiwal, c1, c2, Xte, yte, ...
                                                    Xs, ys, alphas(src), ...
                                                    struct('quiet', 1));
            fprintf('DONE!\n')

            % record performance
            tiwal_query_cts(src,:,(s-1) * reps + r) = cumsum(hist_tal(:,1),1);
            tiwal_err_seq(src,:,(s-1) * reps + r) = hist_tal(:,5);

            fprintf('ERRORS: SLE=%.4f, IWALE=%.4f (%4dq), UTE=%.4f, STE=%.4f, TIWALE=%.4f (%4dq)\n', ...
                        sl_err(s), hist_al(end,5), sum(hist_al(:,1)), ...
                        tlu_err(src, s), tls_err(src, s), hist_tal(end,5), sum(hist_tal(:,1)));
            fprintf('\n')
        end
    end
end

save('icdm-sentiment.mat', 'alphas', 'kfolds', 'reps', 'parts', ...
     'c0_iwal', 'c0_tiwal', 'sl_err', 'tlu_err', 'tls_err', 'dadist', 'crisk', ...
     'iwal_query_cts', 'tiwal_query_cts', 'iwal_err_seq', 'tiwal_err_seq');

step = 50;

acc_mean_sl    = mean(1-sl_err);
iwal_queries_mean = mean(iwal_query_cts, 2);
iwal_acc_mean  = mean(1-iwal_err_seq, 2);

close all;
f1 = figure;
hold on;
plot([ 0 budget ], [ budget budget ], 'k--', 'LineWidth', 2)
plot(0:step:budget, iwal_queries_mean(1:step:end), 'k-', 'LineWidth', 3)

f2 = figure;
hold on;
plot([ 0 budget ], [ acc_mean_sl acc_mean_sl ], 'k--', 'LineWidth', 2)
plot(iwal_queries_mean(1:step:end), iwal_acc_mean(1:step:end), 'k-', 'LineWidth', 3)

cs = [ 'r', 'g', 'b' ];
for src=1:numel(Src)
    acc_mean_tlu   = mean(1-tlu_err(src,:));
    acc_mean_tls   = mean(1-tls_err(src,:));
    tiwal_queries_mean = mean(tiwal_query_cts(src,:,:), 3);
    tiwal_acc_mean = mean(1-tiwal_err_seq(src,:,:), 3);

    figure(f1)
    hold on
    plot(0:step:budget, tiwal_queries_mean(1:step:end), sprintf('%s-', cs(src)), 'LineWidth', 3)
    
    figure(f2)
    hold on
    plot([ 0 budget ], [ acc_mean_tlu acc_mean_tlu ], sprintf('%s:', cs(src)), 'LineWidth', 2)
    plot([ 0 budget ], [ acc_mean_tls acc_mean_tls ], sprintf('%s--', cs(src)), 'LineWidth', 2)
    plot(tiwal_queries_mean(1:step:end), tiwal_acc_mean(1:step:end), sprintf('%s-', cs(src)), 'LineWidth', 3)
end

% plot queries vs. points
% title('Queries vs. points')
% xlabel('Points')
% ylabel('Queries')
% legend('Budget', 'Queries (IWAL)', 'Queries (TIWAL)', 'Location', 'NorthWest')

% axis([ 0 qcount 0.45 0.85 ])
% legend('IWAL CAL mean accuracy + 1 std', 'TIWAL CAL mean accuracy + 1 std', 'Source-only TL mean accuracy', 'Target-only SL mean accuracy', 'Source+target TL mean accuracy')
% xlabel('Queries')
% ylabel('Accuracy')