function [R, t] = run_lsgcpd_engine(source, target, w, max_iter)
% MATLAB Engine entry point for LSG-CPD
% Input:
%   source: (N, 3) source point cloud (array)
%   target: (M, 3) target point cloud (array)
%   w: outlier ratio (default 0.5)
%   max_iter: max iterations (default 50)
% Output:
%   R: (3, 3) rotation matrix
%   t: (3, 1) translation vector

if nargin < 3, w = 0.5; end
if nargin < 4, max_iter = 50; end

% Add utility functions path
addpath(fullfile(fileparts(mfilename('fullpath')), 'utility_functions'));

% Convert arrays to pointCloud objects (LSGCPD expects pointCloud)
pc_source = pointCloud(single(source));
pc_target = pointCloud(single(target));

% Call LSGCPD
tform = LSGCPD(pc_source, pc_target, ...
    'outlierRatio', w, ...
    'maxIter', max_iter);

% Extract results
R = gather(tform.Rotation');  % transpose for column-major
t = gather(tform.Translation');

end
