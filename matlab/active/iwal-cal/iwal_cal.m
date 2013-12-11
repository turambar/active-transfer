function [ y, iw, alhist ] = iwal_cal(X, y, oracle, budget, base_learner, ...
                                      c0, c1, c2, Xte, yte, params)
% IWAL_CAL(X, Y, ORACLE, BUDGET, BASE_LEARNER, C0, C1, C2, XTE, YTE, PARAMS) 
% Wrapper function that runs IWAL_CAL_MAIN with the following:
%
% * weights function:           @simple_weights
% * Gbound function:            @iwal_cal_gbound
% * query probability function: @iwal_cal_query_probability
%
% Together this achieves normal IWAL CAL behavior (vs. TIWAL CAL).
%
% INPUT
%   X               NxD matrix of features; training data
%   y               Nx1 vector of labels; NaN entries indicate unlabeled
%   oracle          Nx1 vector of labels, used as "oracle"
%   budget          label budget
%   baes_learner    base learner function from base-learners directory
%   c0, c1, c2      IWAL CAL parameters
%   Xte             TxD matrix of features; test data; may be empty
%   yte             Tx1 vector of labels; test data; may be empty
%   params          parameters struct
%       .quiet      controls verbosity of IWAL_CAL_MAIN
%
% RETURNS
%   y               Nx1 vector of queried labels
%   iw              Nx1 vector of importance weights
%   alhist          Nx5 matrix of active learning history. The (t)th row
%                   contains: (1) 1/0 whether (t)th label was queried; (2)
%                   Gbound_t, (3) G_t, (4) P_t, (5) test set error, if Xte,
%                   yte provided
%
% Author: Dave Kale (dkale@usc.edu)

[ y, iw, alhist ] = iwal_cal_main(X, y, oracle, budget, base_learner, ...
                                  @simple_weights, ...
                                  @iwal_cal_gbound, ...
                                  @iwal_cal_query_probability, ...
                                  c0, c1, c2, Xte, yte, params);

% Add dummy entry for "0 queries."
alhist = [ 0 nan nan nan 0.5; alhist ];

end