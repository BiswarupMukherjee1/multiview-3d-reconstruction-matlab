%% =====================================================================
%  3D Vision for Multiple and Moving Cameras (3DVMMC) - 2025/2026
%  Lab Evaluation - SECTION 2: Feature matching between views
%  Universidad Autonoma de Madrid
% =====================================================================

clear; close all; clc;

%% ====================== CONFIGURATION ===============================
% --- Paths ---
addpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials\Lab2');   % homography_solve_vmmc, etc.
addpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials');         % vgg_gui_F.m (directly here)

scene_folder   = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Scene_Images';
output_folder  = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Section2_Outputs';
if ~exist(output_folder, 'dir'), mkdir(output_folder); end


vlfeat_setup_script = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials\Lab4\toolboxes\vlfeat-0.9.21\toolbox\vl_setup.m';
USE_VLFEAT_COMPARISON = true;

% --- Image resizing for speed ---
% whatever scale use here, A from Section 1 must be scaled
% by the same factor for Section 3. We'll handle that in Section 3.
RESIZE_FACTOR = 0.5;   % 0.5 = 2000x1500 (recommended), 1.0 = full 4000x3000 (slow)

% --- Pairs of views to study ---
% Mix of short-baseline and longer-baseline pairs.
pairs = [ 1 2;     % very small baseline
          1 3;     % small baseline
          2 4;     % small-medium baseline
          3 6;     % medium baseline
          1 5;     % medium baseline
          2 7;     % medium-large baseline
          4 8 ];   % large baseline

% --- SIFT matching parameters ---
sift_match_ratio   = 0.7;     % Lowe's NNDR threshold (0.6=strict, 0.8=loose)
fund_max_distance  = 1.5;     % RANSAC threshold for F (pixels in resized image)
fund_confidence    = 99.99;   % RANSAC confidence
homog_max_distance = 4.0;     % RANSAC threshold for H

%% ====================== LOAD IMAGES =================================
fprintf('\n============================================================\n');
fprintf('  SECTION 2: Loading scene images\n');
fprintf('============================================================\n');

img_files = dir(fullfile(scene_folder, 'scene_*.jpg'));
N = numel(img_files);
if N < 3, error('Need at least 3 scene images in %s', scene_folder); end
fprintf('Found %d scene images.\n', N);

imgs       = cell(1, N);
imgs_gray  = cell(1, N);
img_sizes  = zeros(N, 2);
for k = 1:N
    I = imread(fullfile(scene_folder, img_files(k).name));
    if RESIZE_FACTOR ~= 1.0
        I = imresize(I, RESIZE_FACTOR);
    end
    imgs{k}      = I;
    imgs_gray{k} = rgb2gray(I);
    img_sizes(k,:) = [size(I,2), size(I,1)];
    fprintf('  Loaded %s -> %d x %d\n', img_files(k).name, size(I,2), size(I,1));
end

%% ====================== MONTAGE (Section 2.a) =======================
fprintf('\nBuilding montage of all views...\n');
fig_montage = figure('Name','Scene Montage','Position',[100 100 1200 800]);
montage_size = [ceil(N/3) 3];   % auto-pick grid
montage(imgs, 'Size', montage_size);
title(sprintf('Scene captured from %d viewpoints', N), 'FontSize', 14);
saveas(fig_montage, fullfile(output_folder, 'montage_all_views.png'));
fprintf('  Saved montage to %s\n', fullfile(output_folder, 'montage_all_views.png'));

%% ====================== SIFT (MATLAB) ON ALL IMAGES =================
fprintf('\n--- Running MATLAB SIFT on all %d images ---\n', N);
kp_matlab   = cell(1, N);
desc_matlab = cell(1, N);

t0 = tic;
for k = 1:N
    pts = detectSIFTFeatures(imgs_gray{k});
    [d, vpts] = extractFeatures(imgs_gray{k}, pts, 'Method', 'SIFT');
    kp_matlab{k}   = vpts;
    desc_matlab{k} = d;
    fprintf('  Image %d: %d SIFT keypoints detected\n', k, size(d,1));
end
fprintf('  Total MATLAB-SIFT time: %.1f s\n', toc(t0));

% Visualize keypoints on each image
fig_kp = figure('Name','Detected SIFT keypoints','Position',[100 100 1400 900]);
ncols = 3; nrows = ceil(N/ncols);
for k = 1:N
    subplot(nrows, ncols, k);
    imshow(imgs{k}); hold on;
    nshow = min(200, length(kp_matlab{k}));
    if nshow > 0
        plot(kp_matlab{k}.selectStrongest(nshow), 'showOrientation', false);
    end
    title(sprintf('Img %d: %d KPs (showing top %d)', k, length(kp_matlab{k}), nshow));
end
saveas(fig_kp, fullfile(output_folder, 'sift_keypoints_all_views.png'));

%% ====================== PROCESS EACH PAIR ===========================
n_pairs = size(pairs, 1);
results = struct([]);

fprintf('\n============================================================\n');
fprintf('  Processing %d view pairs (MATLAB SIFT)\n', n_pairs);
fprintf('============================================================\n');

for p = 1:n_pairs
    i = pairs(p,1); j = pairs(p,2);
    if i > N || j > N
        fprintf('\nSkipping pair %d-%d (image index out of range, N=%d)\n', i, j, N);
        continue;
    end
    fprintf('\n--- Pair %d: views %d <-> %d ---\n', p, i, j);

    % --- Match descriptors with NNDR (Lowe's ratio) ---
    idx_pairs = matchFeatures(desc_matlab{i}, desc_matlab{j}, ...
                              'MatchThreshold', 100, ...
                              'MaxRatio', sift_match_ratio, ...
                              'Unique', true);
    n_putative = size(idx_pairs, 1);
    if n_putative < 8
        fprintf('  Too few putative matches (%d). Skipping pair.\n', n_putative);
        continue;
    end
    pts_i = kp_matlab{i}(idx_pairs(:,1));
    pts_j = kp_matlab{j}(idx_pairs(:,2));
    fprintf('  Putative matches: %d\n', n_putative);

    % --- Estimate Fundamental matrix with RANSAC ---
    [F, inliers_F] = estimateFundamentalMatrix(pts_i.Location, pts_j.Location, ...
        'Method', 'RANSAC', ...
        'NumTrials', 5000, ...
        'DistanceThreshold', fund_max_distance, ...
        'Confidence', fund_confidence);
    n_inl_F = sum(inliers_F);
    inlier_ratio_F = n_inl_F / n_putative;
    fprintf('  F-matrix inliers: %d / %d (%.1f%%)\n', n_inl_F, n_putative, 100*inlier_ratio_F);

    % --- Estimate Homography with RANSAC ---
    H = []; n_inl_H = 0; inlier_ratio_H = 0;
    try
        tform = estimateGeometricTransform2D(pts_i.Location, pts_j.Location, ...
            'projective', ...
            'MaxNumTrials', 5000, ...
            'Confidence', 99.9, ...
            'MaxDistance', homog_max_distance);
        H = tform.T';
        % Count inliers under H by residuals
        p_i_h = [pts_i.Location, ones(n_putative,1)]';
        p_j_h = [pts_j.Location, ones(n_putative,1)]';
        proj  = H * p_i_h;
        proj  = proj ./ proj(3,:);
        res   = sqrt(sum((proj(1:2,:) - p_j_h(1:2,:)).^2, 1));
        inliers_H = res' < homog_max_distance;
        n_inl_H = sum(inliers_H);
        inlier_ratio_H = n_inl_H / n_putative;
        fprintf('  H-matrix inliers: %d / %d (%.1f%%)\n', n_inl_H, n_putative, 100*inlier_ratio_H);
    catch ME
        fprintf('  H-matrix estimation FAILED: %s\n', ME.message);
    end

    % --- Save matches overlay figure (visible briefly so saveas works) ---
    fig_m = figure('Name', sprintf('Matches %d-%d', i, j), 'Position',[100 100 1600 600]);
    showMatchedFeatures(imgs{i}, imgs{j}, pts_i(inliers_F), pts_j(inliers_F), 'montage');
    title(sprintf('Pair %d-%d: F inliers=%d/%d (%.1f%%)', i, j, n_inl_F, n_putative, 100*inlier_ratio_F));
    drawnow;
    saveas(fig_m, fullfile(output_folder, sprintf('pair_%02d_%02d_matches.png', i, j)));
    close(fig_m);

    % --- Store ---
    r.pair_idx        = [i j];
    r.n_putative      = n_putative;
    r.n_inl_F         = n_inl_F;
    r.inlier_ratio_F  = inlier_ratio_F;
    r.F               = F;
    r.n_inl_H         = n_inl_H;
    r.inlier_ratio_H  = inlier_ratio_H;
    r.H               = H;
    r.pts_i_inliers   = pts_i(inliers_F);
    r.pts_j_inliers   = pts_j(inliers_F);
    if isempty(results), results = r; else, results(end+1) = r; end %#ok<SAGROW>
end

%% ====================== SUMMARY TABLE ===============================
fprintf('\n============================================================\n');
fprintf('  SUMMARY (MATLAB SIFT)\n');
fprintf('============================================================\n');
fprintf('%-10s %-12s %-12s %-10s %-12s %-10s\n', ...
        'Pair', '#Putative', '#F-inliers', 'F-ratio', '#H-inliers', 'H-ratio');
for p = 1:numel(results)
    r = results(p);
    fprintf('%-10s %-12d %-12d %-10.1f%% %-12d %-10.1f%%\n', ...
        sprintf('%d-%d', r.pair_idx(1), r.pair_idx(2)), ...
        r.n_putative, r.n_inl_F, 100*r.inlier_ratio_F, ...
        r.n_inl_H, 100*r.inlier_ratio_H);
end

%% ====================== PICK BEST PAIR ==============================
n_inl_vec = arrayfun(@(r) r.n_inl_F, results);
[~, best] = max(n_inl_vec);
best_r = results(best);
i_b = best_r.pair_idx(1); j_b = best_r.pair_idx(2);

fprintf('\nBest pair: %d-%d  (#F-inliers=%d)\n', i_b, j_b, best_r.n_inl_F);

%% ====================== PANORAMA (best pair) ========================
if ~isempty(best_r.H)
    fprintf('\nBuilding panorama for the best pair (%d-%d)...\n', i_b, j_b);
    try
        tformBest = projective2d(best_r.H');
        [xlim_i, ylim_i] = outputLimits(tformBest, [1 size(imgs{i_b},2)], [1 size(imgs{i_b},1)]);
        xMin = min(1, xlim_i(1));
        xMax = max(size(imgs{j_b},2), xlim_i(2));
        yMin = min(1, ylim_i(1));
        yMax = max(size(imgs{j_b},1), ylim_i(2));
        width  = round(xMax - xMin);
        height = round(yMax - yMin);

        xLim = [xMin xMax]; yLim = [yMin yMax];
        panoramaView = imref2d([height width], xLim, yLim);

        warped_i = imwarp(imgs{i_b}, tformBest, 'OutputView', panoramaView);
        warped_j = imwarp(imgs{j_b}, projective2d(eye(3)), 'OutputView', panoramaView);

        % Simple max-blend (works without vision.AlphaBlender)
        panorama = max(warped_i, warped_j);

        fig_pan = figure('Name','Panorama (best pair)','Position',[100 100 1600 600]);
        imshow(panorama);
        title(sprintf('Panorama from views %d and %d', i_b, j_b));
        drawnow;
        saveas(fig_pan, fullfile(output_folder, sprintf('panorama_pair_%02d_%02d.png', i_b, j_b)));
        fprintf('  Panorama saved.\n');
    catch ME
        fprintf('  Panorama generation FAILED: %s\n', ME.message);
    end
end

%% ====================== VLFEAT COMPARISON (best pair) ===============
vlfeat_summary = '(skipped)';
if USE_VLFEAT_COMPARISON
    fprintf('\n--- VLFeat SIFT comparison on best pair (%d-%d) ---\n', i_b, j_b);
    try
        run(vlfeat_setup_script);
        Ia = single(imgs_gray{i_b});
        Ib = single(imgs_gray{j_b});
        [fa, da] = vl_sift(Ia);
        [fb, db] = vl_sift(Ib);
        fprintf('  VLFeat KPs: img%d=%d, img%d=%d\n', i_b, size(fa,2), j_b, size(fb,2));

        [matches_v, ~] = vl_ubcmatch(da, db, 2.0);
        n_v = size(matches_v, 2);
        fprintf('  VLFeat putative matches: %d\n', n_v);

        if n_v >= 8
            pa = fa(1:2, matches_v(1,:))';
            pb = fb(1:2, matches_v(2,:))';
            [~, inl_v] = estimateFundamentalMatrix(pa, pb, ...
                'Method','RANSAC', ...
                'NumTrials', 5000, ...
                'DistanceThreshold', fund_max_distance, ...
                'Confidence', fund_confidence);
            n_inl_v = sum(inl_v);
            fprintf('  VLFeat F-inliers: %d / %d (%.1f%%)\n', n_inl_v, n_v, 100*n_inl_v/n_v);
            vlfeat_summary = sprintf(['VLFeat: %d KPs (img%d), %d KPs (img%d), ', ...
                                      '%d putative, %d F-inliers (%.1f%%)'], ...
                size(fa,2), i_b, size(fb,2), j_b, n_v, n_inl_v, 100*n_inl_v/n_v);
        end
    catch ME
        fprintf('  VLFeat run FAILED: %s\n', ME.message);
        fprintf('  (This is OK -- MATLAB SIFT result still holds for the report.)\n');
        vlfeat_summary = sprintf('VLFeat failed: %s', ME.message);
    end
end

%% ====================== SAVE RESULTS ================================
save(fullfile(output_folder, 'Section2_results.mat'), ...
     'results', 'best_r', 'pairs', 'RESIZE_FACTOR', 'vlfeat_summary', ...
     'img_files', 'img_sizes');

% Plain-text summary for the report
fid = fopen(fullfile(output_folder, 'Section2_summary.txt'), 'w');
fprintf(fid, 'Section 2 summary  (RESIZE_FACTOR = %.2f)\n\n', RESIZE_FACTOR);
fprintf(fid, 'Image sizes used for SIFT: %d x %d\n\n', img_sizes(1,1), img_sizes(1,2));
fprintf(fid, '%-10s %-12s %-12s %-10s %-12s %-10s\n', ...
        'Pair', '#Putative', '#F-inliers', 'F-ratio', '#H-inliers', 'H-ratio');
for p = 1:numel(results)
    r = results(p);
    fprintf(fid, '%-10s %-12d %-12d %-9.1f%% %-12d %-9.1f%%\n', ...
        sprintf('%d-%d', r.pair_idx(1), r.pair_idx(2)), ...
        r.n_putative, r.n_inl_F, 100*r.inlier_ratio_F, ...
        r.n_inl_H, 100*r.inlier_ratio_H);
end
fprintf(fid, '\nBest pair: %d-%d  (highest F-inliers)\n', i_b, j_b);
fprintf(fid, '\nBest-pair Fundamental matrix F (image %d <-> image %d):\n', i_b, j_b);
for r = 1:3
    fprintf(fid, '  %12.6e  %12.6e  %12.6e\n', best_r.F(r,1), best_r.F(r,2), best_r.F(r,3));
end
fprintf(fid, '\nBest-pair Homography matrix H (image %d -> image %d):\n', i_b, j_b);
if ~isempty(best_r.H)
    for r = 1:3
        fprintf(fid, '  %12.6e  %12.6e  %12.6e\n', best_r.H(r,1), best_r.H(r,2), best_r.H(r,3));
    end
end
fprintf(fid, '\n%s\n', vlfeat_summary);
fclose(fid);
fprintf('\nText summary written to Section2_summary.txt\n');

%% ====================== OPEN vgg_gui_F ON BEST PAIR =================
fprintf('\n--- Opening vgg_gui_F on the best pair for visual inspection ---\n');
fprintf('   (Click points in either image to see epipolar lines.)\n');
fprintf('   (Take 2-3 screenshots for the report.)\n');
try
    vgg_gui_F(imgs{i_b}, imgs{j_b}, best_r.F);
catch ME
    fprintf('  vgg_gui_F FAILED: %s\n', ME.message);
    fprintf('  Make sure vgg_gui_F.m is on the MATLAB path.\n');
end

fprintf('\nSection 2 done. Outputs in %s\n', output_folder);