function [ dl, dr ] = approx_da_distance(Xs, Xt, varargin)
% APPROX_DA_DISTANCE(Xs, Xt, varargin) Calculates an approximation to "dA"
% distance, as in Blitzer, et al's NIPS 2008 paper. Otherwise known as the
% "domain separator hypothesis."
%
% INPUT
%   Xs          NxD matrix of source features
%   Xt          NxD matrix of target features
%   varargin    allows us to customize the behavior by passing in a
%               different learning function, etc.
%               1:  learner (from base-learners directory)
%
% RETURNS
%   hl          average hinge loss per example for SVM
%   hr          average 0-1 loss per example for SVM
%   ll          average logistic loss per example for logistic regression
%   lr          average 0-1 loss per example for logistic regression
% 
% This function calculates an approximation to the "dA" distance between two
% data domains, as in Blitzer, Crammer, Kulezsa, Pereira, and Wortman,
% "Learning Bounds for Domain Adaptation," NIPS 2008. Intuitively, dA
% distance quantifies the "separability" (as in the binary classificatin
% sense) of two domains. The more "separable" two data sets are, the
% greater the dA distance and the "less similar" they are. If it's tough to
% separate two data sets, then the lower the dA distance will be. dA
% distance depends on the hypothesis class, but it can be computed with
% only unlabeled samples from the domains, which is convenient.
%
% Blitzer, et al., approximate dA distance by "training a linear classifier
% to discriminate between the two domains. We use a standard hinge loss
% (normalized by the number of instances) and apply the quantity
% 1 - (hinge loss)." Here we try two things: we train an L2 regularized SVM
% and compute hinge loss, as Blitzer, et al., do, and we also train a
% regularized logistic regression and report the average logistic loss.
% These two values are often pretty close but not always. For good measure,
% we also report a average 0-1 loss for each of these classifiers. One
% thing that Blitzer, et al., do not make clear is whether they do a
% train/test split or whether they simply compute performance on the
% "training" data. We suspect the latter and so follow that.
%
% dA distance actually depends on the hypothesis class so we now provide
% the option to pass in an arbitrary learner (via the varargin parameters)
% and use it to approximate dA distance as well.
%
% NOTE: default behavior trains weighted instance linear SVM using the
% liblinear library modified to accept instance weights
% (http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances).
% When MEXing liblinear, be sure to rename the train and predict functions
% to lltrain and llpredict.
%
% Author: Dave Kale (dkale@usc.edu)

X = sparse([ Xs; Xt ]);
m = size(Xs,1);
n = size(Xt,1);
ws = (m + n) / m;
wt = (m + n) / n;
w = [ ws * ones(m,1); wt * ones(n,1) ];
y = [ ones(m,1); -ones(n,1) ];

model = lltrain(w, y, X, sprintf('-s 3 -c 1 -B 1 -q'));
[ yhat, ~, ydv ] = llpredict(y, X, model, '-q');
if model.Label(1) < 0
    yhat = -yhat;
    ydv = -ydv;
end
dl = 1 - (sum(max(0, 1 - y .* ydv) / length(y)));
dr = sum(y~=yhat) / length(y);

if numel(varargin) > 1
    learner = varargin{1};
    params  = varargin{2};
    h = learner(X, y, w, params);
    [ yhat, ydv ] = h(X);
    dl = 1 - (sum(max(0, 1 - y .* ydv) / length(y)));
    dr = sum(y~=yhat) / length(y);
end

model = lltrain(w, y, X, sprintf('-s 0 -c 1 -B 1 -q'));
[ yhat, ~, ydv ] = llpredict(y, X, model, '-q'); %'-b 1');
if model.Label(1) < 0
    yhat = -yhat;
    ydv = -ydv;
end
ll = sum(log(1 + exp(-y .* ydv))) / length(y);
lr = sum(y~=yhat) / length(y);

end
