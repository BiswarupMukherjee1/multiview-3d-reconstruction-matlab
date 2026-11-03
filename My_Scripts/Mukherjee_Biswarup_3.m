%% =====================================================================
%  3D Vision for Multiple and Moving Cameras (3DVMMC) - 2025/2026
%  Lab Evaluation - SECTION 3: Multi-view 3D reconstruction (v2 - robust)
%  Universidad Autonoma de Madrid
% =====================================================================

clear; close all; clc;

%% ====================== CONFIGURATION ===============================
addpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials\Lab2');
addpath(genpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials\Lab6'));
addpath(genpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials\vgg_allfns'));
addpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials');

scene_folder      = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Scene_Images';
output_folder     = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Section3_Outputs_v3';
section1_matfile  = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\My_Scripts\Section1_PartA_screen.mat';

if ~exist(output_folder, 'dir'), mkdir(output_folder); end

RESIZE_FACTOR     = 0.5;
sift_match_ratio  = 0.7;
fund_max_distance = 1.5;
min_track_length  = 3;

% --- Initial pair ---
init_view_a = 2;
init_view_b = 7;

% --- ROBUSTNESS THRESHOLDS ---
MIN_POINTS_PER_CAMERA       = 200;  % drop cameras with fewer track obs
RESECTION_ERROR_THRESHOLD   = 5.0;  % px - obs with bigger error get pruned before BA
FINAL_EUC_ERROR_THRESHOLD   = 10.0; % px - prune after Euclidean resection

ransac_F_trials   = 5000;
ransac_F_conf     = 99.99;

fprintf('\n============================================================\n');
fprintf('  SECTION 3 (v2): Multi-view 3D reconstruction with robustness\n');
fprintf('============================================================\n');

%% ====================== LOAD A MATRIX ===============================
fprintf('\n--- Loading intrinsic matrix A from Section 1 ---\n');
S1 = load(section1_matfile);
A_full = S1.A_screen;

A = A_full;
A(1,1) = A_full(1,1) * RESIZE_FACTOR;
A(1,2) = A_full(1,2) * RESIZE_FACTOR;
A(1,3) = A_full(1,3) * RESIZE_FACTOR;
A(2,2) = A_full(2,2) * RESIZE_FACTOR;
A(2,3) = A_full(2,3) * RESIZE_FACTOR;
K = A;
fprintf('K (scaled):\n'); disp(K);

%% ====================== LOAD IMAGES =================================
img_files = dir(fullfile(scene_folder, 'scene_*.jpg'));
N = numel(img_files);
imgs = cell(1,N); imgs_gray = cell(1,N);
for k = 1:N
    I = imread(fullfile(scene_folder, img_files(k).name));
    if RESIZE_FACTOR ~= 1.0
        I = imresize(I, RESIZE_FACTOR);
    end
    imgs{k} = I; imgs_gray{k} = rgb2gray(I);
end
fprintf('Loaded %d images (%dx%d).\n', N, size(imgs{1},2), size(imgs{1},1));

%% ====================== SIFT + PAIRWISE MATCHING ====================
fprintf('\n--- SIFT detection ---\n');
kp = cell(1,N); desc = cell(1,N);
for k = 1:N
    pts = detectSIFTFeatures(imgs_gray{k});
    [d, vpts] = extractFeatures(imgs_gray{k}, pts, 'Method','SIFT');
    kp{k} = vpts; desc{k} = d;
    fprintf('  Image %d: %d KPs\n', k, length(vpts));
end

fprintf('\n--- Pairwise matching ---\n');
pairwise = cell(N,N);
for i = 1:N
    for j = i+1:N
        idx = matchFeatures(desc{i}, desc{j}, ...
            'MatchThreshold',100, 'MaxRatio',sift_match_ratio, 'Unique',true);
        if size(idx,1) < 8, pairwise{i,j} = zeros(0,2); continue; end
        pts_i = kp{i}(idx(:,1)).Location;
        pts_j = kp{j}(idx(:,2)).Location;
        try
            [~, inl] = estimateFundamentalMatrix(pts_i, pts_j, ...
                'Method','RANSAC','NumTrials',ransac_F_trials, ...
                'DistanceThreshold',fund_max_distance,'Confidence',ransac_F_conf);
            pairwise{i,j} = idx(inl,:);
        catch
            pairwise{i,j} = zeros(0,2);
        end
        fprintf('  Pair %d-%d: %d inliers\n', i, j, size(pairwise{i,j},1));
    end
end

%% ====================== N-VIEW TRACKS (union-find) ==================
fprintf('\n--- Building N-view tracks ---\n');
nkp = cellfun(@length, kp);
offset = [0, cumsum(nkp(:)')];
total_nodes = offset(end);
parent = (1:total_nodes)';
rank_ = zeros(total_nodes,1);
gid = @(img,k) offset(img) + k;

for i = 1:N
    for j = i+1:N
        m = pairwise{i,j};
        for r = 1:size(m,1)
            a = gid(i, m(r,1)); b = gid(j, m(r,2));
            ra = a; while parent(ra)~=ra, parent(ra)=parent(parent(ra)); ra=parent(ra); end
            rb = b; while parent(rb)~=rb, parent(rb)=parent(parent(rb)); rb=parent(rb); end
            if ra ~= rb
                if rank_(ra)<rank_(rb), parent(ra)=rb;
                elseif rank_(ra)>rank_(rb), parent(rb)=ra;
                else, parent(rb)=ra; rank_(ra)=rank_(ra)+1; end
            end
        end
    end
end
for x = 1:total_nodes
    r0 = x; while parent(r0)~=r0, r0=parent(r0); end
    parent(x) = r0;
end
[~, ~, group] = unique(parent);
n_groups = max(group);

tracks = cell(n_groups,1);
for x = 1:total_nodes, tracks{group(x)}(end+1) = x; end

kept_tracks = {};
for g = 1:n_groups
    nodes = tracks{g};
    if length(nodes) < min_track_length, continue; end
    img_idx = zeros(length(nodes),1); kp_idx_in = zeros(length(nodes),1);
    for r = 1:length(nodes)
        node = nodes(r);
        for ii = 1:N
            if node>offset(ii) && node<=offset(ii+1)
                img_idx(r) = ii;
                kp_idx_in(r) = node - offset(ii);
                break;
            end
        end
    end
    uimg = unique(img_idx);
    if length(uimg) < min_track_length, continue; end
    if length(uimg) ~= length(img_idx), continue; end
    t.img_idx = img_idx; t.kp_idx = kp_idx_in;
    kept_tracks{end+1} = t;
end
n_tracks = numel(kept_tracks);
fprintf('Total connected components: %d. Filtered tracks (>=%d views): %d\n', ...
        n_groups, min_track_length, n_tracks);

x_all = NaN(3, n_tracks, N);
vp = zeros(n_tracks, N);
for t_i = 1:n_tracks
    tr = kept_tracks{t_i};
    for r = 1:length(tr.img_idx)
        im = tr.img_idx(r); ki = tr.kp_idx(r);
        loc = kp{im}(ki).Location;
        x_all(1,t_i,im) = loc(1);
        x_all(2,t_i,im) = loc(2);
        x_all(3,t_i,im) = 1;
        vp(t_i,im) = 1;
    end
end

% Save track-points visualization (Step 1.a)
fig_kp = figure('Name','N-view tracks','Position',[100 100 1400 900]);
ncols=3; nrows=ceil(N/ncols);
for k = 1:N
    subplot(nrows,ncols,k); imshow(imgs{k}); hold on;
    vis = vp(:,k)==1;
    pts2d = squeeze(x_all(1:2,vis,k))';
    if ~isempty(pts2d), plot(pts2d(:,1),pts2d(:,2),'g.','MarkerSize',6); end
    title(sprintf('Img %d: %d track pts', k, sum(vis)));
end
sgtitle(sprintf('N-view track points (kept in \\geq %d views)', min_track_length));
saveas(fig_kp, fullfile(output_folder,'nview_tracks_per_image.png'));

%% ====================== STEP 2: INITIAL 2-VIEW RECON ================
fprintf('\n============================================================\n');
fprintf('  STEP 2: Initial projective recon from cameras %d-%d\n', init_view_a, init_view_b);
fprintf('============================================================\n');
mask_init = (vp(:,init_view_a)==1) & (vp(:,init_view_b)==1);
fprintf('Tracks visible in both: %d\n', sum(mask_init));

x_init = x_all(:, mask_init, [init_view_a, init_view_b]);
pts_a = squeeze(x_init(1:2,:,1))';
pts_b = squeeze(x_init(1:2,:,2))';

[F_init, inl_init] = estimateFundamentalMatrix(pts_a, pts_b, ...
    'Method','RANSAC','NumTrials',ransac_F_trials, ...
    'DistanceThreshold',fund_max_distance,'Confidence',ransac_F_conf);
fprintf('F RANSAC: %d / %d inliers (%.1f%%)\n', ...
        sum(inl_init), sum(mask_init), 100*sum(inl_init)/sum(mask_init));

idx_init = find(mask_init);
keep_global = idx_init(inl_init);
x_keep = x_all(:, keep_global, :);
vp_keep = vp(keep_global, :);
n_kept = size(x_keep, 2);
fprintf('After F-RANSAC filtering, %d tracks kept.\n', n_kept);

% Canonical P from F
[~,~,V] = svd(F_init');
e_prime = V(:,3); e_prime = e_prime/e_prime(3);
P1 = [eye(3), zeros(3,1)];
P2 = [Cross2Matrix(e_prime)*F_init, e_prime];

x_for_tri = x_keep(:,:,[init_view_a, init_view_b]);
[X3d_init, ~] = linear_triangulation(x_for_tri, cat(3,P1,P2));
fprintf('Initial 3D points triangulated: %d\n', size(X3d_init,2));

err_2view = compute_per_obs_error(x_keep(:,:,[init_view_a,init_view_b]), ...
                                  cat(3,P1,P2), X3d_init);
fprintf('\nStep 2.b: 2-view reproj error  mean=%.3f  median=%.3f  std=%.3f  n=%d\n', ...
        mean(err_2view), median(err_2view), std(err_2view), numel(err_2view));
save_error_histogram(err_2view, 'Step 2.b: 2-view reprojection error', ...
    fullfile(output_folder,'step2b_hist_2view_reproj_error.png'));

%% ====================== STEP 3: ROBUST CAMERA SELECTION =============
fprintf('\n============================================================\n');
fprintf('  STEP 3: Camera selection, resectioning, then BA\n');
fprintf('============================================================\n');

% Count visible track-obs per camera (excluding init cameras, which are P1/P2 by construction)
vis_count = sum(vp_keep, 1);
fprintf('Visible tracks per camera (out of %d):\n', n_kept);
for k = 1:N
    fprintf('  Camera %d: %d  %s\n', k, vis_count(k), ...
        ternary(vis_count(k)>=MIN_POINTS_PER_CAMERA, '[USE]', '[DROP - too few]'));
end

% Always include init cameras. For the others, only if they have enough.
selected = false(1,N);
selected(init_view_a) = true;
selected(init_view_b) = true;
for k = 1:N
    if k==init_view_a || k==init_view_b, continue; end
    if vis_count(k) >= MIN_POINTS_PER_CAMERA
        selected(k) = true;
    end
end
sel_idx = find(selected);
n_sel = length(sel_idx);
fprintf('Selected %d cameras for reconstruction: %s\n', n_sel, mat2str(sel_idx));

x_sel = x_keep(:,:,sel_idx);
vp_sel = vp_keep(:,sel_idx);
N_sel = length(sel_idx);

% Resection each non-initial selected camera
pos_a_sel = find(sel_idx == init_view_a);
pos_b_sel = find(sel_idx == init_view_b);

P_resect = zeros(3,4,N_sel);
P_resect(:,:,pos_a_sel) = P1;
P_resect(:,:,pos_b_sel) = P2;

for k = 1:N_sel
    if k == pos_a_sel || k == pos_b_sel, continue; end
    vis = vp_sel(:,k)==1;
    x_k = x_sel(:,vis,k);
    X_k = X3d_init(:,vis);
    try
        P_resect(:,:,k) = PDLT_NA(x_k, X_k);
        % Check resection quality immediately
        proj = P_resect(:,:,k) * X_k;
        proj = proj ./ proj(3,:);
        d = sqrt(sum((proj(1:2,:) - x_k(1:2,:)).^2,1));
        fprintf('  Cam idx %d (sel pos %d): resectioned, median err = %.2f px\n', ...
                sel_idx(k), k, median(d));
    catch ME
        fprintf('  Cam idx %d: resection FAILED (%s)\n', sel_idx(k), ME.message);
    end
end

err_resect_all = compute_per_obs_error_with_vp(x_sel, P_resect, X3d_init, vp_sel);
fprintf('\nStep 3.a (i): reproj err after resectioning ALL kept cameras\n');
fprintf('  Mean: %.3f  Median: %.3f  Std: %.3f  n=%d\n', ...
        mean(err_resect_all), median(err_resect_all), std(err_resect_all), numel(err_resect_all));
save_error_histogram(err_resect_all, ...
    'Step 3.a (i): reprojection error after resectioning', ...
    fullfile(output_folder,'step3a_i_hist_resection_error.png'));

%% ====================== NO PRE-BA PRUNE (v3) ========================
% do NOT prune outliers before BA.
% BA is robust to outliers via its non-linear least squares formulation.
% Pruning before BA kills good observations whose projective-frame
% reprojection looked bad only because of the projective frame distortion.
fprintf('\n--- Skipping pre-BA prune (v3 design - let BA handle outliers) ---\n');

x_clean    = x_sel;
vp_clean2  = vp_sel;
X3d_pre_BA = X3d_init;
fprintf('Feeding BA: %d tracks, %d observations\n', ...
        size(x_clean,2), sum(vp_clean2(:)));

%% ====================== BUNDLE ADJUSTMENT ===========================
fprintf('\n--- Running Projective Bundle Adjustment ---\n');
t0 = tic;
[P_BA, X3d_BA] = BAProjectiveCalib(x_clean, P_resect, X3d_pre_BA, vp_clean2);
fprintf('  BA time: %.1fs\n', toc(t0));

err_BA = compute_per_obs_error_with_vp(x_clean, P_BA, X3d_BA, vp_clean2);
fprintf('\nStep 3.a (ii): reproj err after BA\n');
fprintf('  Mean: %.3f  Median: %.3f  Std: %.3f  n=%d\n', ...
        mean(err_BA), median(err_BA), std(err_BA), numel(err_BA));
save_error_histogram(err_BA, 'Step 3.a (ii): reproj error after Projective BA', ...
    fullfile(output_folder,'step3a_ii_hist_BA_error.png'));

%% ====================== STEP 4: F FROM BA P's =======================
fprintf('\n============================================================\n');
fprintf('  STEP 4: F from BA projection matrices\n');
fprintf('============================================================\n');
P_a_BA = P_BA(:,:,pos_a_sel);
P_b_BA = P_BA(:,:,pos_b_sel);
F_from_P = vgg_F_from_P({P_a_BA, P_b_BA});
F_from_P = F_from_P / norm(F_from_P,'fro');
fprintf('F from BA P matrices (normalized):\n'); disp(F_from_P);

%% ====================== STEP 5: EUCLIDEAN UPGRADE ===================
fprintf('\n============================================================\n');
fprintf('  STEP 5: Essential matrix and Euclidean reconstruction\n');
fprintf('  (using re-projected points after BA, per PDF Step 5)\n');
fprintf('============================================================\n');
E = K' * F_from_P * K;
fprintf('E:\n'); disp(E);

[R_two, T_one] = factorize_E(E);
combos = {R_two(:,:,1), T_one; R_two(:,:,1), -T_one; ...
          R_two(:,:,2), T_one; R_two(:,:,2), -T_one};

% Compute BA re-projected points (q_rep_BA) in cameras a and b.
% These are the "noise-free" 2D points consistent with BA's refined geometry.
% use the re-projected points obtained after
% the Projective Bundle Adjustment step.
fprintf('\nComputing BA re-projected points in cameras %d and %d...\n', init_view_a, init_view_b);
q_rep_a = P_BA(:,:,pos_a_sel) * X3d_BA;
q_rep_a = q_rep_a ./ q_rep_a(3,:);   % normalize to pixel coords (homog with last=1)
q_rep_b = P_BA(:,:,pos_b_sel) * X3d_BA;
q_rep_b = q_rep_b ./ q_rep_b(3,:);
x_rep_BA = zeros(3, size(X3d_BA,2), 2);
x_rep_BA(:,:,1) = q_rep_a;
x_rep_BA(:,:,2) = q_rep_b;

% Tracks visible in BOTH initial cameras after BA filtering
both_vis = (vp_clean2(:,pos_a_sel)==1) & (vp_clean2(:,pos_b_sel)==1);
fprintf('Tracks visible in both initial cameras after BA: %d\n', sum(both_vis));

P1_euc = K * [eye(3), zeros(3,1)];
best_count = -1;

for c = 1:4
    R = combos{c,1};  T = combos{c,2};
    P2_test = K * [R, T];
    % Triangulate using BA RE-PROJECTED points
    [X_try, ~] = linear_triangulation(x_rep_BA(:,both_vis,:), cat(3, P1_euc, P2_test));
    Xun = X_try ./ X_try(4,:);
    depth1 = Xun(3,:);
    Xc2 = R * Xun(1:3,:) + T;
    depth2 = Xc2(3,:);
    n_pos = sum(depth1>0 & depth2>0);
    sign_t = '+'; if mod(c,2)==0, sign_t='-'; end
    fprintf('  Combo %d (R%d, %sT): %d / %d in front\n', ...
            c, 1+(c>2), sign_t, n_pos, sum(both_vis));
    if n_pos > best_count
        best_count = n_pos; best_combo = c;
        best_R = R; best_T = T;
    end
end
fprintf('Selected combo %d (in-front count: %d)\n', best_combo, best_count);

% Final Euclidean triangulation using BA re-projected points
P2_euc = K * [best_R, best_T];
[X3d_euc, ~] = linear_triangulation(x_rep_BA(:,both_vis,:), cat(3, P1_euc, P2_euc));
keep_euc_local = find(both_vis);

%% ====================== STEP 6: EUCLIDEAN RESECT ALL CAMS ===========
fprintf('\n============================================================\n');
fprintf('  STEP 6: Resection all cameras in Euclidean frame\n');
fprintf('============================================================\n');

P_euc_all = zeros(3,4,N_sel);
P_euc_all(:,:,pos_a_sel) = P1_euc;
P_euc_all(:,:,pos_b_sel) = P2_euc;

x_for_resect_euc = x_clean(:, keep_euc_local, :);
vp_for_resect_euc = vp_clean2(keep_euc_local, :);
CAM_MEDIAN_REJECT_PX = 5.0;   % v3: drop cameras with median err > this after Euclidean resection

for k = 1:N_sel
    if k == pos_a_sel || k == pos_b_sel, continue; end
    vis = vp_for_resect_euc(:,k)==1;
    if sum(vis) < 30
        fprintf('  Cam idx %d: only %d points - SKIP\n', sel_idx(k), sum(vis));
        continue;
    end
    x_k = x_for_resect_euc(:,vis,k);
    X_k = X3d_euc(:,vis);
    try
        P_k = PDLT_NA(x_k, X_k);
        % Verify quality on visible-to-this-camera points
        proj = P_k * X_k; proj = proj ./ proj(3,:);
        d = sqrt(sum((proj(1:2,:) - x_k(1:2,:)).^2,1));
        if median(d) > CAM_MEDIAN_REJECT_PX
            fprintf('  Cam idx %d: median err %.2f px > %.1f - DROPPING this camera\n', ...
                    sel_idx(k), median(d), CAM_MEDIAN_REJECT_PX);
            continue;
        end
        P_euc_all(:,:,k) = P_k;
        fprintf('  Cam idx %d: Euclidean resection OK (median err %.2f px, %d points)\n', ...
                sel_idx(k), median(d), sum(vis));
    catch ME
        fprintf('  Cam idx %d: FAILED (%s)\n', sel_idx(k), ME.message);
    end
end

% Final prune of outlier observations (only for clean visualization)
fprintf('\n--- Final pruning of Euclidean outliers (threshold = %.1f px) ---\n', FINAL_EUC_ERROR_THRESHOLD);
vp_final = vp_for_resect_euc;
n_pruned2 = 0;
for k = 1:N_sel
    if all(P_euc_all(:,:,k)==0,'all')
        % Camera was dropped - zero out its visibility
        vp_final(:,k) = 0;
        continue;
    end
    vis = vp_for_resect_euc(:,k)==1;
    proj = P_euc_all(:,:,k) * X3d_euc;
    proj = proj ./ proj(3,:);
    d = sqrt(sum((proj(1:2,:) - x_for_resect_euc(1:2,:,k)).^2,1))';
    bad = vis & (d > FINAL_EUC_ERROR_THRESHOLD);
    vp_final(bad,k) = 0;
    n_pruned2 = n_pruned2 + sum(bad);
end
fprintf('Pruned %d additional observations (in surviving cameras).\n', n_pruned2);

err_euc = compute_per_obs_error_with_vp(x_for_resect_euc, P_euc_all, X3d_euc, vp_final);
fprintf('\nStep 6.a: final Euclidean reproj error\n');
fprintf('  Mean: %.3f  Median: %.3f  Std: %.3f  n=%d\n', ...
        mean(err_euc), median(err_euc), std(err_euc), numel(err_euc));
save_error_histogram(err_euc, 'Step 6.a: final Euclidean reprojection error', ...
    fullfile(output_folder,'step6a_hist_euclidean_error.png'));

%% ====================== 3D VISUALIZATION ============================
fprintf('\n--- Step 6.b: 3D visualization ---\n');

% v3: Keep tracks visible in >= 2 surviving cameras
final_track_visibility = sum(vp_final, 2);
keep_for_viz = final_track_visibility >= 2;
X3d_viz = X3d_euc(:, keep_for_viz);
fprintf('Visualizing %d / %d 3D points (after final prune).\n', ...
        size(X3d_viz, 2), size(X3d_euc, 2));

% If still very few, relax the constraint
if size(X3d_viz, 2) < 100
    fprintf('  Fewer than 100 points to visualize - relaxing to all valid Euclidean points\n');
    % Use all triangulated Euclidean points whose depth is positive
    X_un_check = X3d_euc ./ X3d_euc(4,:);
    valid_depth = X_un_check(3,:) > 0 & X_un_check(3,:) < 100;
    X3d_viz = X3d_euc(:, valid_depth);
    keep_for_viz = valid_depth(:);
    fprintf('  Now visualizing %d points\n', size(X3d_viz, 2));
end

% Decompose P_euc into R, T_center per camera
K_per = repmat(K, [1,1,N_sel]);
R_per = zeros(3,3,N_sel);
T_per = zeros(3, N_sel);
valid_cams = false(1,N_sel);
for k = 1:N_sel
    Pk = P_euc_all(:,:,k);
    if all(Pk(:)==0), continue; end
    Rt = K \ Pk;
    R_per(:,:,k) = Rt(:,1:3);
    t_tr = Rt(:,4);
    T_per(:,k) = -R_per(:,:,k)' * t_tr;   % camera center in world coords
    valid_cams(k) = true;
end
K_per_v = K_per(:,:,valid_cams);
R_per_v = R_per(:,:,valid_cams);
T_per_v = T_per(:,valid_cams);

fig_3d_cam = figure('Name','3D point cloud + cameras', 'Position',[100 100 1100 800]);
draw_scene(X3d_viz, K_per_v, R_per_v, T_per_v);
title('Euclidean reconstruction: 3D points + camera frustums');
xlabel('X'); ylabel('Y'); zlabel('Z'); view(45,30);
saveas(fig_3d_cam, fullfile(output_folder,'step6b_pointcloud_with_cameras_view1.png'));
view(-30,20); saveas(fig_3d_cam, fullfile(output_folder,'step6b_pointcloud_with_cameras_view2.png'));
view(0,90);   saveas(fig_3d_cam, fullfile(output_folder,'step6b_pointcloud_with_cameras_topdown.png'));

X_un = X3d_viz ./ X3d_viz(4,:);
fig_only = figure('Name','3D point cloud (no cameras)','Position',[100 100 1100 800]);
scatter3(X_un(1,:), X_un(2,:), X_un(3,:), 8, 'b', 'filled');
daspect([1 1 1]); axis vis3d; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Euclidean 3D point cloud (cameras hidden)');
view(45,30); saveas(fig_only, fullfile(output_folder,'step6b_pointcloud_only_view1.png'));
view(-30,20); saveas(fig_only, fullfile(output_folder,'step6b_pointcloud_only_view2.png'));

%% ====================== STEP 6.c: RGB COLOURED ======================
fprintf('\n--- Step 6.c: RGB-coloured point cloud ---\n');
ref_image = imgs{init_view_a};
n_viz = size(X3d_viz, 2);
colors_rgb = zeros(n_viz, 3);
viz_track_idx = find(keep_for_viz);
for t_i = 1:n_viz
    g = viz_track_idx(t_i);
    if vp_for_resect_euc(g, pos_a_sel) == 0
        colors_rgb(t_i,:) = [0.5 0.5 0.5];
        continue;
    end
    px = round(x_for_resect_euc(1, g, pos_a_sel));
    py = round(x_for_resect_euc(2, g, pos_a_sel));
    px = max(1, min(px, size(ref_image,2)));
    py = max(1, min(py, size(ref_image,1)));
    rgb = ref_image(py, px, :);
    colors_rgb(t_i,:) = double(rgb(:)')/255;
end

fig_rgb = figure('Name','3D point cloud (RGB coloured)','Position',[100 100 1100 800]);
scatter3(X_un(1,:), X_un(2,:), X_un(3,:), 14, colors_rgb, 'filled');
daspect([1 1 1]); axis vis3d; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title(sprintf('Euclidean 3D point cloud, RGB sampled from image %d', init_view_a));
view(45,30); saveas(fig_rgb, fullfile(output_folder,'step6c_pointcloud_rgb_view1.png'));
view(-30,20); saveas(fig_rgb, fullfile(output_folder,'step6c_pointcloud_rgb_view2.png'));
view(0,90);   saveas(fig_rgb, fullfile(output_folder,'step6c_pointcloud_rgb_topdown.png'));

%% ====================== SAVE EVERYTHING =============================
fprintf('\n--- Saving results ---\n');
save(fullfile(output_folder,'Section3_results.mat'), ...
     'A','K','RESIZE_FACTOR','init_view_a','init_view_b','sel_idx','selected', ...
     'F_init','P1','P2','X3d_init', ...
     'P_BA','X3d_BA','vp_clean2','x_clean', ...
     'F_from_P','E','best_R','best_T', ...
     'P_euc_all','X3d_euc','vp_final','colors_rgb', ...
     'err_2view','err_resect_all','err_BA','err_euc', ...
     '-v7.3');

fid = fopen(fullfile(output_folder,'Section3_summary.txt'),'w');
fprintf(fid,'Section 3 summary (v2 - robust)\n===============================\n\n');
fprintf(fid,'Image resize factor: %.2f (working at %dx%d)\n', ...
        RESIZE_FACTOR, round(4000*RESIZE_FACTOR), round(3000*RESIZE_FACTOR));
fprintf(fid,'Intrinsic K (scaled):\n');
for r=1:3, fprintf(fid,'  %12.4f %12.4f %12.4f\n', K(r,1),K(r,2),K(r,3)); end
fprintf(fid,'\nN-view tracks: %d (visible in >=%d views)\n', n_tracks, min_track_length);
fprintf(fid,'Tracks kept after F-RANSAC on pair %d-%d: %d\n', init_view_a, init_view_b, n_kept);
fprintf(fid,'Cameras selected for reconstruction: %s\n', mat2str(sel_idx));
%fprintf(fid,'Tracks after outlier-prune + 3-view filter: %d\n', sum(keep_tracks_post_prune));
fprintf(fid,'Tracks in Euclidean reconstruction: %d\n', sum(both_vis));
fprintf(fid,'Tracks visualized: %d\n', size(X3d_viz,2));
fprintf(fid,'\nReprojection errors (px):\n');
fprintf(fid,'  2.b   (2-view initial):     mean=%.3f  median=%.3f  std=%.3f  n=%d\n', ...
        mean(err_2view),median(err_2view),std(err_2view),numel(err_2view));
fprintf(fid,'  3.a(i) (after resectioning): mean=%.3f  median=%.3f  std=%.3f  n=%d\n', ...
        mean(err_resect_all),median(err_resect_all),std(err_resect_all),numel(err_resect_all));
fprintf(fid,'  3.a(ii) (after BA):          mean=%.3f  median=%.3f  std=%.3f  n=%d\n', ...
        mean(err_BA),median(err_BA),std(err_BA),numel(err_BA));
fprintf(fid,'  6.a   (Euclidean final):    mean=%.3f  median=%.3f  std=%.3f  n=%d\n', ...
        mean(err_euc),median(err_euc),std(err_euc),numel(err_euc));
fprintf(fid,'\nE decomposition: selected combo %d / 4 (in-front count %d)\n', best_combo, best_count);
fprintf(fid,'Best R:\n');
for r=1:3, fprintf(fid,'  %8.4f %8.4f %8.4f\n', best_R(r,1),best_R(r,2),best_R(r,3)); end
fprintf(fid,'Best T direction: [%.4f %.4f %.4f]\n', best_T(1),best_T(2),best_T(3));
fclose(fid);

fprintf('\n============================================================\n');
fprintf('  SECTION 3 (v2) DONE.  Output: %s\n', output_folder);
fprintf('============================================================\n');

%% ====================== LOCAL HELPER FUNCTIONS ======================
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function err = compute_per_obs_error(x, P, X)
    [~, npts, ncam] = size(x);
    err = [];
    for k = 1:ncam
        proj = P(:,:,k) * X;
        proj = proj ./ proj(3,:);
        d = sqrt(sum((proj(1:2,:) - x(1:2,:,k)).^2,1));
        err = [err, d];
    end
end

function err = compute_per_obs_error_with_vp(x, P, X, vp)
    [~, npts, ncam] = size(x);
    err = [];
    for k = 1:ncam
        if all(P(:,:,k)==0,'all'), continue; end
        vis = (vp(:,k)==1);
        if ~any(vis), continue; end
        proj = P(:,:,k) * X(:,vis);
        proj = proj ./ proj(3,:);
        d = sqrt(sum((proj(1:2,:) - x(1:2,vis,k)).^2,1));
        err = [err, d];
    end
end

function save_error_histogram(err, ttl, filepath)
    fig = figure('Visible','on','Position',[100 100 800 500]);
    histogram(err, 50, 'Normalization','count', 'FaceColor',[0.2 0.5 0.8]);
    xlabel('Reprojection error (pixels)');
    ylabel('Count');
    title(sprintf('%s\nmean=%.3f, median=%.3f, n=%d', ...
                  ttl, mean(err), median(err), numel(err)));
    grid on; drawnow;
    saveas(fig, filepath);
    close(fig);
end