function r = approx_combined_risk(Xs, ys, Xt, yt, varargin)
% APPROX_COMBINED_RISK(Xs, ys, Xt, yt, varargin) Calculates an approximation to the
% combined risk, as defined in Blitzer, et al's NIPS 2008 paper.
%
% INPUT
%   X1          N x D matrix of source features
%   y1          N x 1 vector of source labels (+1 positive, -1 negative)
%   X2          N x D matrix of target features
%   y2          N x 1 vector of target labels (+1 positive, -1 negative)
%   varargin    allows us to customize the behavior by passing in a
%               different learning function, etc.
%               1:  learner (from base-learners directory)
%
% RETURNS
%   r           approximate combined risk
% 
% This function calculates an approximation to the "combined risk" concept,
% as defined in Blitzer, Crammer, Kulezsa, Pereira, and Wortman, "Learning
% Bounds for Domain Adaptation," NIPS 2008:
%
%       risk_C(h) = risk_S(h) + risk_T(h)
%
% The "ideal" hypothesis h* minimizes this combined risk:
%
%       h* = argmin_{h \in H} risk_S(h) + risk_T(h)
%
% In the Blitzer paper, the risk_C(h*) is a constant source of error for
% their approach to transfer learning.
%
% NOTE: default behavior trains weighted instance linear SVM using the
% liblinear library modified to accept instance weights
% (http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances).
% When MEXing liblinear, be sure to rename the train and predict functions
% to lltrain and llpredict.
%
% Author: Dave Kale (dkale@usc.edu)

K = 10;
parts = data_kfold(ys, K, 1);
partt = data_kfold(yt, K, 0);
risks = zeros(1, K);

if numel(varargin) > 1
    learner = varargin{1};
    params  = varargin{2};
else
    learner = @svm_linearl2l2;
    params  = struct();
end

for k=1:K
    Xtr = [ Xs(parts~=k,:); Xt(partt~=k,:) ];
    ytr = [ ys(parts~=k); yt(partt~=k) ];
    m = sum(parts~=k);
    n = sum(partt~=k);
    ws = (m + n) / m;
    wt = (m + n) / n;
    wdom = [ ws * ones(m,1); wt * ones(n,1) ];
    wlab = zeros(size(ytr));
    wlab(ytr>0) = length(ytr) / sum(ytr>0);
    wlab(ytr<0) = length(ytr) / sum(ytr<0);
    wtr = wdom .* wlab;
    
    Xte = [ Xs(parts==k,:); Xt(partt==k,:) ];
    yte = [ ys(parts==k); yt(partt==k) ];

    h = learner(Xtr, ytr, wtr, params);
    [ yhat, ~ ] = h(Xte);
    risks(k) = sum(yhat~=yte) / length(yte);
end

r = mean(risks);

end
