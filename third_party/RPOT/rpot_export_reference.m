function rpot_export_reference(out_path)
% Export MATLAB RPOT result for strict numerical comparison with PyTorch.
%
% Usage (from repo root):
%   matlab -batch "addpath('compare_methods/RPOT'); rpot_export_reference('compare_methods/RPOT/matlab_ref.mat');"

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);

    if nargin < 1
        out_path = fullfile(this_dir, 'matlab_reference.mat');
    end

    data = load(fullfile(this_dir, 'data.mat'));
    X = data.X;
    Y = data.Y;

    [pointCountX, x_dim] = size(X);
    [pointCountY, ~] = size(Y);
    Mx = 1 / pointCountX * ones(pointCountX, 1);
    My = 1 / pointCountY * ones(pointCountY, 1);

    % algorithm params (same as UnbalanceRegistration.m)
    para.epsilon = 0.004;
    para.alpha = 0;
    para.beta = 1;
    para.alpha_totalmass = 0;
    para.beta_totalmass = 0.8;
    para.threhold = 1e-5;

    if x_dim == 2
        para.AnnealRate = 0.8;
    else
        para.AnnealRate = 0.9;
    end

    Xorg = X;
    Yorg = Y;

    % preprocess data, align the mass barycenter between X and Y
    XmassBarycenter = sum(1 / pointCountX * X);
    YmassBarycenter = sum(1 / pointCountY * Y);

    X = bsxfun(@minus, X, XmassBarycenter);
    Y = bsxfun(@minus, Y, YmassBarycenter);

    % compute transport distance matrix between X and Y
    D = pdist2(X, Y, 'squaredeuclidean');

    [R0, t0, Ytransformed, D, T, para] = unbalanced_OT(X, Y, Mx, My, D, para);

    t0 = t0 + XmassBarycenter - YmassBarycenter * R0;
    Yorgtransformed = bsxfun(@plus, Yorg * R0, t0);

    save(out_path, 'R0', 't0', 'Yorgtransformed');
end

