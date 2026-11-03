%% =====================================================================
%  3D Vision for Multiple and Moving Cameras (3DVMMC) - 2025/2026
%  Lab Evaluation - SECTION 2 (extension): per-octave filtering experiment
%                                           and best-4-pairs selection
%  Universidad Autonoma de Madrid
% =====================================================================

clear; close all; clc;

%% ====================== CONFIGURATION ===============================
addpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials\Lab2');
addpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials');   % vgg_gui_F

scene_folder   = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Scene_Images';
output_folder  = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Section2_Outputs';

% Same parameters as the main script
RESIZE_FACTOR      = 0.5;
sift_match_ratio   = 0.7;
fund_max_distance  = 1.5;
fund_confidence    = 99.99;

% SIFT octave bounds (MATLAB default base sigma is 1.6)
SIFT_BASE_SIGMA   = 1.6;
N_OCTAVES_TO_TEST = 4;   % octaves 0,1,2,3

%% ====================== LOAD SAVED RESULTS ==========================
fprintf('\n============================================================\n');
fprintf('  SECTION 2 EXTENSION: per-octave + best-4-pairs\n');
fprintf('============================================================\n');

mat_path = fullfile(output_folder, 'Section2_results.mat');
if ~exist(mat_path, 'file')
    error('Cannot find %s. Run Surname_Name_2.m first.', mat_path);
end
S = load(mat_path);
results = S.results;
n_pairs = numel(results);
fprintf('Loaded %d pair results from previous run.\n', n_pairs);

%% ====================== TOP-4 PAIRS SELECTION =======================
n_inl_vec = arrayfun(@(r) r.n_inl_F, results);
[~, order] = sort(n_inl_vec, 'descend');
top4_idx = order(1:min(4, numel(order)));

fprintf('\n--- Top 4 pairs by F-inlier count ---\n');
fprintf('%-6s %-10s %-12s %-12s %-10s\n', 'Rank', 'Pair', '#Putative', '#F-inliers', 'F-ratio');
for k = 1:numel(top4_idx)
    r = results(top4_idx(k));
    fprintf('%-6d %-10s %-12d %-12d %-9.1f%%\n', k, ...
        sprintf('%d-%d', r.pair_idx(1), r.pair_idx(2)), ...
        r.n_putative, r.n_inl_F, 100*r.inlier_ratio_F);
end

%% ====================== RELOAD IMAGES ===============================
img_files = dir(fullfile(scene_folder, 'scene_*.jpg'));
N = numel(img_files);
imgs       = cell(1, N);
imgs_gray  = cell(1, N);
for k = 1:N
    I = imread(fullfile(scene_folder, img_files(k).name));
    if RESIZE_FACTOR ~= 1.0
        I = imresize(I, RESIZE_FACTOR);
    end
    imgs{k}      = I;
    imgs_gray{k} = rgb2gray(I);
end
fprintf('\nReloaded %d images.\n', N);

%% ====================== RE-DETECT SIFT (with octave info) ===========
fprintf('\n--- Re-running MATLAB SIFT (with octave bookkeeping) ---\n');
kp_all   = cell(1, N);
desc_all = cell(1, N);
oct_all  = cell(1, N);   % per-keypoint octave index (0,1,2,...)

t0 = tic;
for k = 1:N
    pts = detectSIFTFeatures(imgs_gray{k});
    [d, vpts] = extractFeatures(imgs_gray{k}, pts, 'Method', 'SIFT');
    kp_all{k}   = vpts;
    desc_all{k} = d;
    % Octave from scale: octave = floor(log2(scale / base_sigma))
    s = vpts.Scale;
    oct_all{k} = max(0, floor(log2(s / SIFT_BASE_SIGMA)));
    fprintf('  Image %d: %d KPs.  Octave histogram (0,1,2,3,4+): %s\n', ...
        k, length(vpts), mat2str(histcounts(oct_all{k}, [0 1 2 3 4 inf])));
end
fprintf('  Total time: %.1f s\n', toc(t0));

% Sanity check: print scale ranges per octave (using best pair's first image)
i_b = results(top4_idx(1)).pair_idx(1);
j_b = results(top4_idx(1)).pair_idx(2);
fprintf('\nSanity check (scale ranges in image %d):\n', i_b);
for oct = 0:(N_OCTAVES_TO_TEST-1)
    mask = (oct_all{i_b} == oct);
    if any(mask)
        ss = kp_all{i_b}.Scale(mask);
        fprintf('  Octave %d: %d KPs, scale range [%.2f, %.2f]\n', ...
                oct, sum(mask), min(ss), max(ss));
    else
        fprintf('  Octave %d: empty\n', oct);
    end
end

%% ====================== PER-OCTAVE EXPERIMENT (on best pair) ========
fprintf('\n============================================================\n');
fprintf('  PER-OCTAVE EXPERIMENT on best pair %d-%d\n', i_b, j_b);
fprintf('============================================================\n');
fprintf('Note: octave 0 has many keypoints, matching may take 30-60s.\n');

octave_results = struct([]);
for oct = 0:(N_OCTAVES_TO_TEST-1)
    mask_i = (oct_all{i_b} == oct);
    mask_j = (oct_all{j_b} == oct);
    n_i = sum(mask_i); n_j = sum(mask_j);
    fprintf('\n--- Octave %d: %d KPs in img%d, %d KPs in img%d ---\n', ...
        oct, n_i, i_b, n_j, j_b);

    % Default empty record
    r = struct('octave', oct, 'n_i', n_i, 'n_j', n_j, ...
               'n_putative', 0, 'n_inl_F', 0, 'inlier_ratio_F', 0, 'F', []);

    if n_i < 8 || n_j < 8
        fprintf('  Too few keypoints in this octave. Skipping.\n');
        if isempty(octave_results), octave_results = r; else, octave_results(end+1) = r; end %#ok<SAGROW>
        continue;
    end

    kp_i = kp_all{i_b}(mask_i);
    kp_j = kp_all{j_b}(mask_j);
    d_i  = desc_all{i_b}(mask_i, :);
    d_j  = desc_all{j_b}(mask_j, :);

    % Match
    t1 = tic;
    idx_pairs = matchFeatures(d_i, d_j, ...
        'MatchThreshold', 100, ...
        'MaxRatio', sift_match_ratio, ...
        'Unique', true);
    n_put = size(idx_pairs, 1);
    r.n_putative = n_put;
    fprintf('  Putative matches: %d   (match time: %.1fs)\n', n_put, toc(t1));

    if n_put < 8
        fprintf('  Too few putative matches.\n');
        if isempty(octave_results), octave_results = r; else, octave_results(end+1) = r; end %#ok<SAGROW>
        continue;
    end

    pts_i = kp_i(idx_pairs(:,1));
    pts_j = kp_j(idx_pairs(:,2));

    % Estimate F with RANSAC
    inl = false(n_put, 1);
    try
        [F, inl] = estimateFundamentalMatrix(pts_i.Location, pts_j.Location, ...
            'Method', 'RANSAC', ...
            'NumTrials', 5000, ...
            'DistanceThreshold', fund_max_distance, ...
            'Confidence', fund_confidence);
        r.F             = F;
        r.n_inl_F       = sum(inl);
        r.inlier_ratio_F = r.n_inl_F / n_put;
        fprintf('  F-inliers: %d / %d (%.1f%%)\n', r.n_inl_F, n_put, 100*r.inlier_ratio_F);
    catch ME
        fprintf('  F estimation FAILED: %s\n', ME.message);
    end

    % Save figure with matches overlay (if we have enough inliers)
    if r.n_inl_F >= 8
        fig_oct = figure('Name', sprintf('Octave %d matches', oct), ...
                         'Position', [50 50 1600 600]);
        showMatchedFeatures(imgs{i_b}, imgs{j_b}, pts_i(inl), pts_j(inl), 'montage');
        title(sprintf('Pair %d-%d, octave %d only: %d/%d inliers (%.1f%%)', ...
              i_b, j_b, oct, r.n_inl_F, n_put, 100*r.inlier_ratio_F));
        drawnow;
        saveas(fig_oct, fullfile(output_folder, ...
            sprintf('octave%d_pair_%02d_%02d_matches.png', oct, i_b, j_b)));
        close(fig_oct);
    end

    if isempty(octave_results), octave_results = r; else, octave_results(end+1) = r; end %#ok<SAGROW>
end

%% ====================== PER-OCTAVE SUMMARY ==========================
fprintf('\n============================================================\n');
fprintf('  PER-OCTAVE SUMMARY (best pair %d-%d)\n', i_b, j_b);
fprintf('============================================================\n');
fprintf('%-8s %-12s %-12s %-12s %-12s %-10s\n', ...
        'Octave', sprintf('#KPs img%d', i_b), sprintf('#KPs img%d', j_b), ...
        '#Putative', '#F-inliers', 'F-ratio');
for k = 1:numel(octave_results)
    r = octave_results(k);
    fprintf('%-8d %-12d %-12d %-12d %-12d %-9.1f%%\n', ...
        r.octave, r.n_i, r.n_j, r.n_putative, r.n_inl_F, 100*r.inlier_ratio_F);
end

% Reference: ALL octaves combined (from main script)
ref = results(top4_idx(1));
fprintf('%-8s %-12d %-12d %-12d %-12d %-9.1f%%\n', 'ALL', ...
        length(kp_all{i_b}), length(kp_all{j_b}), ...
        ref.n_putative, ref.n_inl_F, 100*ref.inlier_ratio_F);

%% ====================== BAR CHART: octaves ==========================
fig_bar = figure('Name','Per-octave inlier counts','Position',[100 100 900 500]);
octs = arrayfun(@(r) r.octave, octave_results);
puts = arrayfun(@(r) r.n_putative, octave_results);
inls = arrayfun(@(r) r.n_inl_F, octave_results);
hb = bar([puts; inls]', 'grouped');
xlabel('SIFT octave (only keypoints from that octave used)');
ylabel('Number of matches');
legend({'Putative matches', 'F-inlier matches'}, 'Location', 'best');
title(sprintf('Per-octave matching performance, pair %d-%d', i_b, j_b));
set(gca, 'XTick', 1:numel(octs));
set(gca, 'XTickLabel', arrayfun(@(o) sprintf('Oct %d', o), octs, 'UniformOutput', false));
grid on;
saveas(fig_bar, fullfile(output_folder, ...
    sprintf('per_octave_barchart_pair_%02d_%02d.png', i_b, j_b)));

%% ====================== SAVE TEXT SUMMARY ===========================
fid = fopen(fullfile(output_folder, 'Section2_extension_summary.txt'), 'w');
fprintf(fid, 'Section 2 EXTENSION summary\n');
fprintf(fid, '===========================\n\n');

fprintf(fid, '--- Top 4 pairs by F-inlier count ---\n');
fprintf(fid, '%-6s %-10s %-12s %-12s %-10s\n', 'Rank', 'Pair', '#Putative', '#F-inliers', 'F-ratio');
for k = 1:numel(top4_idx)
    r = results(top4_idx(k));
    fprintf(fid, '%-6d %-10s %-12d %-12d %-9.1f%%\n', k, ...
        sprintf('%d-%d', r.pair_idx(1), r.pair_idx(2)), ...
        r.n_putative, r.n_inl_F, 100*r.inlier_ratio_F);
end

fprintf(fid, '\n--- Per-octave experiment on best pair %d-%d ---\n', i_b, j_b);
fprintf(fid, '%-8s %-12s %-12s %-12s %-12s %-10s\n', ...
        'Octave', sprintf('#KPs img%d', i_b), sprintf('#KPs img%d', j_b), ...
        '#Putative', '#F-inliers', 'F-ratio');
for k = 1:numel(octave_results)
    r = octave_results(k);
    fprintf(fid, '%-8d %-12d %-12d %-12d %-12d %-9.1f%%\n', ...
        r.octave, r.n_i, r.n_j, r.n_putative, r.n_inl_F, 100*r.inlier_ratio_F);
end
fprintf(fid, '%-8s %-12d %-12d %-12d %-12d %-9.1f%%\n', 'ALL', ...
        length(kp_all{i_b}), length(kp_all{j_b}), ...
        ref.n_putative, ref.n_inl_F, 100*ref.inlier_ratio_F);

fprintf(fid, '\nSIFT base sigma assumed: %.2f\n', SIFT_BASE_SIGMA);
fprintf(fid, 'Octaves tested: 0 to %d\n', N_OCTAVES_TO_TEST-1);
fclose(fid);

%% ====================== SAVE .MAT ===================================
save(fullfile(output_folder, 'Section2_extension.mat'), ...
     'octave_results', 'top4_idx', 'results', 'SIFT_BASE_SIGMA', ...
     'N_OCTAVES_TO_TEST', 'i_b', 'j_b');

fprintf('\nExtension done.\n');
fprintf('  Text summary: %s\n', fullfile(output_folder, 'Section2_extension_summary.txt'));
fprintf('  Bar chart:    %s\n', fullfile(output_folder, sprintf('per_octave_barchart_pair_%02d_%02d.png', i_b, j_b)));
fprintf('  Per-octave match images: octave*_pair_*_matches.png in %s\n', output_folder);