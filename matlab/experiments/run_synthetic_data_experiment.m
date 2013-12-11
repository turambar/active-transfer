% Script to run experiments with 2-feature synthetic data
clear; clc; close all;

%% General setup
m = 512;        % source points
n = 1024;       % target points
reps = 20;     % repetitions
budget = n;     % query budget
base_learner = @svm_linearl2l2;
c0_iwal = 4;
c1 = 1; %5 + 2*sqrt(2);
c2 = 1; %5;
c0_tiwal = 4;
alp = 0.5;

dadist  = zeros(1, reps);
crisk   = zeros(1, reps);

sl_err  = zeros(1, reps);
tlu_err = zeros(1, reps);
tls_err = zeros(1, reps);

iwal_err  = zeros(1, reps);
tiwal_err = zeros(1, reps);

iwal_query_cts  = zeros(budget+1, reps);
tiwal_query_cts = zeros(budget+1, reps);
iwal_err_seq    = zeros(budget+1, reps);
tiwal_err_seq   = zeros(budget+1, reps);

for r=1:reps
    fprintf('rep %3d...\n', r)
    
    %% Generate synthetic 2-feature data
    % source data
    Xs = sparse([ bsxfun(@plus, bsxfun(@times, rand(3*m/8, 2), [ 2 2 ]), [  0  0 ]);
                  bsxfun(@plus, bsxfun(@times, rand(  m/8, 2), [ 2 2 ]), [  0 -1 ]);
                  bsxfun(@plus, bsxfun(@times, rand(3*m/8, 2), [ 2 2 ]), [ -2  0 ]);
                  bsxfun(@plus, bsxfun(@times, rand(  m/8, 2), [ 2 2 ]), [ -2 -1 ]) ]);
    ys = [ ones(m/2,1); -ones(m/2,1) ];

    % training data
    Sigma = chol([2 1; 1 4]*4/3);
    Xtr = [ repmat([ 1  2],n/2,1) + randn(n/2,2) * Sigma;
            repmat([-2 -1],n/2,1) + randn(n/2,2) * Sigma ];
    Xtr = bsxfun(@rdivide, bsxfun(@minus, Xtr, mean(Xtr,1)), std(Xtr,1));
    ytr = [ ones(n/2,1); -ones(n/2,1) ];

    % test data
    Xte = [ repmat([ 1  2],n/2,1) + randn(n/2,2) * Sigma;
            repmat([-2 -1],n/2,1) + randn(n/2,2) * Sigma ];
    Xte = bsxfun(@rdivide, bsxfun(@minus, Xte, mean(Xte,1)), std(Xte,1));
    yte = [ ones(n/2,1); -ones(n/2,1) ];
    
    dadist(r) = approx_da_distance(Xs, Xtr);
    fprintf('DA distance   = %.4f\n', dadist(r));
    crisk(r)  = approx_combined_risk(Xs, ys, Xtr, ytr);
    fprintf('Combined risk = %.4f\n', crisk(r));

    %% shuffle training data
    query_idx = randperm(n);
    one_of_each = [ find(ytr(query_idx)>0, 1, 'first') find(ytr(query_idx)<0, 1, 'first') ];
    query_idx = [ query_idx(one_of_each) query_idx(setdiff(1:n, one_of_each)) ];
    [ ~, sort_idx ] = sort(query_idx);

    Xq = Xtr(query_idx,:);
    yq = ytr(query_idx);

    %% test error, fully supervised
    [ h, ~ ] = base_learner(Xq, yq, ones(size(yq)), struct());
    [ yhat, ~ ] = h(Xte);
    sl_err(r) = err(yte, yhat);

    %% test error, unsupervised transfer
    ws = tiwal_cal_weights(ones(size(ys)), m, struct('m', m, 'alpha', alp));
    [ h, ~ ] = base_learner(Xs, ys, ws, struct());
    [ yhat, ~ ] = h(Xte);
    tlu_err(r) = err(yte, yhat);

    %% test error, fully supervised transfer
    X = [ Xs; Xq ];
    y = [ ys; yq ];
    w = tiwal_cal_weights(ones(size(y)), m + n, struct('m', m, 'alpha', alp));
    [ h, ~ ] = base_learner(X, y, w, struct());
    [ yhat, ~ ] = h(Xte);
    tls_err(r) = err(yte, yhat);

    fprintf('ERRORS: sup=%.4f, unsup transfer=%.4f, sup transfer=%.4f\n', ...
    sl_err(r), tlu_err(r), tls_err(r));
    fprintf('\n')

    %% run IWAL CAL
    fprintf('Run IWAL CAL...\n')
    [ y_al, iw_al, hist_al ] = iwal_cal(Xq, nan(size(yq)), yq, ...
                                        budget, base_learner, ...
                                        c0_iwal, c1, c2, Xte, yte, struct('quiet', 1));
    fprintf('DONE!\n')

    % record performance
    iwal_query_cts(:,r) = cumsum(hist_al(:,1),1);
    iwal_err_seq(:,r) = hist_al(:,5);

    % test error, importance weighted (IWAL)
    idx = ~isnan(y_al);
    [ h, ~ ] = base_learner(Xq(idx,:), yq(idx), iw_al(idx), struct());
    [ yhat, ~ ] = h(Xte);
    iwal_err(r) = err(yte, yhat);
    fprintf('%4d queries, final error=%.4f, IW sup error=%.4f\n', ...
            sum(hist_al(:,1)), hist_al(end,5), iwal_err(r));

    %% run TIWAL CAL
    fprintf('Run TIWAL CAL...\n')
    [ y_tal, iw_tal, hist_tal ] = tiwal_cal(Xq, nan(size(yq)), yq, ...
                                            budget, base_learner, ...
                                            c0_tiwal, c1, c2, Xte, yte, ...
                                            Xs, ys, alp, struct('quiet', 1));
    fprintf('DONE!\n')

    % record performance
    tiwal_query_cts(:,r) = cumsum(hist_tal(:,1),1);
    tiwal_err_seq(:,r) = hist_tal(:,5);

    % test error, importance weighted (TIWAL)
    X = [ Xs; Xq ];
    y = [ ys; y_tal ];
    idx = ~isnan(y);
    w = tiwal_cal_weights([ ones(size(ys)); iw_tal ], m + n, struct('m', m, 'alpha', alp));
    [ h, ~ ] = base_learner(X(idx,:), y(idx), w(idx), struct());
    [ yhat, ~ ] = h(Xte);
    tiwal_err(r) = err(yte, yhat);
    fprintf('%4d queries, final error=%.4f, IW transfer error=%.4f\n', ...
            sum(hist_tal(:,1)), hist_tal(end,5), tiwal_err(r));
    fprintf('\n')
end

iwal_acc_mean      = mean(1-iwal_err_seq, 2);
iwal_acc_std       = std(1-iwal_err_seq, 0, 2);
tiwal_acc_mean     = mean(1-tiwal_err_seq, 2);
tiwal_acc_std      = std(1-tiwal_err_seq, 0, 2);
iwal_queries_mean  = mean(iwal_query_cts, 2);
iwal_queries_std   = std(iwal_query_cts, 0, 2);
tiwal_queries_mean = mean(tiwal_query_cts, 2);
tiwal_queries_std  = std(tiwal_query_cts, 0, 2);

save(sprintf('synthetic-results-forplot-%s.mat', datestr(now, 'YYYYmmDDHHMMSS')), 'iwal_acc_mean', 'iwal_acc_std', ...
     'tiwal_acc_mean', 'tiwal_acc_std', 'iwal_queries_mean', 'iwal_queries_std', ...
     'tiwal_queries_mean', 'tiwal_queries_std', 'budget', 'dadist');

acc_mean_sl    = mean(1-sl_err);
acc_std_sl     = std(1-sl_err);
acc_mean_tlu   = mean(1-tlu_err);
acc_std_tlu    = std(1-tlu_err);;
acc_mean_tls   = mean(1-tls_err);
acc_std_tls    = std(1-tls_err);
acc_mean_iwal  = mean(1-iwal_err);
acc_std_iwal   = std(1-iwal_err);
acc_mean_tiwal = mean(1-tiwal_err);
acc_std_tiwal  = std(1-tiwal_err);

save(sprintf('synthetic-results-raw-%s.mat', datestr(now, 'YYYYmmDDHHMMSS')), 'sl_err', ...
     'tlu_err', 'tls_err', 'iwal_err', 'tiwal_err', 'iwal_query_cts', ...
     'tiwal_query_cts', 'iwal_err_seq', 'tiwal_err_seq', 'dadist');

save(sprintf('synthetic-results-perfs-%s.mat', datestr(now, 'YYYYmmDDHHMMSS')), ...
     'acc_mean_sl', 'acc_std_sl', 'acc_mean_tlu', 'acc_std_tlu', ...
     'acc_mean_tls', 'acc_std_tls', 'acc_mean_iwal', 'acc_std_iwal', ...
     'acc_mean_tiwal', 'acc_std_tiwal', 'dadist');

close all;
figure; hold on;
% plot queries vs. points
plot([ 0 budget ], [ budget budget ], 'k-', 'LineWidth', 3)
plot(0:budget, iwal_queries_mean, 'b-', 'LineWidth', 3)
plot(0:budget, tiwal_queries_mean, 'r-', 'LineWidth', 3)
axis([ 0 budget 0 budget+0.25*budget ])
title('Queries vs. points')
xlabel('Points')
ylabel('Queries')
legend('Budget', 'Queries (IWAL)', 'Queries (TIWAL)', 'Location', 'NorthWest')

qcount = ceil(max(max(iwal_queries_mean), max(tiwal_queries_mean))) + 1;
figure; hold on
fill([ iwal_queries_mean' fliplr(iwal_queries_mean') ], [ iwal_acc_mean'-iwal_acc_std' fliplr(iwal_acc_mean'+iwal_acc_std') ], 'b')
fill([ tiwal_queries_mean' fliplr(tiwal_queries_mean') ], [ tiwal_acc_mean'-tiwal_acc_std' fliplr(tiwal_acc_mean'+tiwal_acc_std') ], 'r')
alpha(0.5)

plot([ 1 qcount ], [ acc_mean_tlu acc_mean_tlu ], 'k--', 'LineWidth', 3)
plot([ 1 qcount ], [ acc_mean_sl acc_mean_sl ], 'b--', 'LineWidth', 3)
plot([ 1 qcount ], [ acc_mean_tls acc_mean_tls ], 'r--', 'LineWidth', 3)
plot(iwal_queries_mean, [ iwal_acc_mean iwal_acc_mean ], 'b-', 'LineWidth', 3)
plot(tiwal_queries_mean, [ tiwal_acc_mean tiwal_acc_mean ], 'r-', 'LineWidth', 3)
axis([ 0 qcount 0.75 1.0 ])
legend('IWAL CAL mean accuracy + 1 std', 'TIWAL CAL mean accuracy + 1 std', 'Source-only TL mean accuracy', 'Target-only SL mean accuracy', 'Source+target TL mean accuracy', 'Location', 'NorthEast')
xlabel('Queries')
ylabel('Accuracy')

figure; hold on
fill([ iwal_queries_mean' fliplr(iwal_queries_mean') ], [ iwal_acc_mean'-iwal_acc_std' fliplr(iwal_acc_mean'+iwal_acc_std') ], 'b')
fill([ tiwal_queries_mean' fliplr(tiwal_queries_mean') ], [ tiwal_acc_mean'-tiwal_acc_std' fliplr(tiwal_acc_mean'+tiwal_acc_std') ], 'r')
alpha(0.5)

plot([ 1 qcount ], [ acc_mean_tlu acc_mean_tlu ], 'k--', 'LineWidth', 3)
plot([ 1 qcount ], [ acc_mean_sl acc_mean_sl ], 'b--', 'LineWidth', 3)
plot([ 1 qcount ], [ acc_mean_tls acc_mean_tls ], 'r--', 'LineWidth', 3)
plot(iwal_queries_mean, [ iwal_acc_mean iwal_acc_mean ], 'b-', 'LineWidth', 3)
plot(tiwal_queries_mean, [ tiwal_acc_mean tiwal_acc_mean ], 'r-', 'LineWidth', 3)
axis([ 0 25 0.75 1.0 ])
legend('IWAL CAL mean accuracy + 1 std', 'TIWAL CAL mean accuracy + 1 std', 'Source-only TL mean accuracy', 'Target-only SL mean accuracy', 'Source+target TL mean accuracy', 'Location', 'NorthEast')
xlabel('Queries')
ylabel('Accuracy')

fprintf('  DA distance: %.4g\t%.4g\n', mean(dadist), std(dadist))
fprintf('Combined risk: %.4g\t%.4g\n', mean(crisk), std(crisk))
fprintf('      Full SL: %.4g\t%.4g\n', mean(sl_err), std(sl_err))
fprintf('     Unsup TL: %.4g\t%.4g\n', mean(tlu_err), std(tlu_err))
fprintf('      Full TL: %.4g\t%.4g\n', mean(tls_err), std(tls_err))
fprintf('         IWAL: %.4g\t%.4g\n', mean(iwal_err), std(iwal_err))
fprintf('        TIWAL: %.4g\t%.4g\n', mean(tiwal_err), std(tiwal_err))