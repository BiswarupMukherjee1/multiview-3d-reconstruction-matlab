%% =====================================================================
%  3D Vision for Multiple and Moving Cameras (3DVMMC)
%  Lab Evaluation - SECTION 1: Camera calibration
%  ---------------------------------------------------------------------
%  SECTION 1 has TWO parts:
%    PART A: calibrate with the screen checkerboard  -> matrix  A
%    PART B: calibrate with your own physical pattern -> matrix  A'
%
% =====================================================================

clear; close all; clc;

%% ====================== CONFIGURATION ===============================
addpath('D:\IPCVAI\SEM2\3D_VMMC_Exam\Lab_Materials\Lab2');

% --- PART A: Screen checkerboard parameters --------------------------
screen_imgs_folder = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Calibration_Images\Screen';
screen_pattern_width_mm = 200;     % 20 cm x 20 cm on screen

% --- PART B: Physical pattern parameters -----------------------------
phys_imgs_folder = 'D:\IPCVAI\SEM2\3D_VMMC_Exam\Calibration_Images\Physical';
phys_pattern_width_mm = 80;        % 80 mm x 80 mm hand-drawn pattern

% --- Common parameters ----------------------------------------------
Np = 9;
show_pattern_order = 1;

% If a saved Part-A result already exists, reuse it instead of re-clicking.
partA_savefile = 'Section1_PartA_screen.mat';


%% ===================================================================
%  PART A: SCREEN CHECKERBOARD CALIBRATION
%% ===================================================================
fprintf('\n============================================================\n');
fprintf('  PART A: Camera calibration with SCREEN checkerboard\n');
fprintf('============================================================\n');

if exist(partA_savefile, 'file') == 2
    fprintf('Found %s -- LOADING saved Part-A results (no re-clicking).\n', ...
            partA_savefile);
    load(partA_savefile);   % brings back A_screen, H_screen, R_screen, etc.
    fprintf('\n--- [SCREEN] A (refined homographies)  [loaded] ---\n');
    disp(A_screen);
else
    fprintf('No saved Part-A results found -- running Part A from scratch.\n');
    [A_screen, A_screen_noref, H_screen, H_screen_noref, R_screen, T_screen, ...
     img_sizes_screen, img_files_screen] = ...
            calibrate_camera_section1(screen_imgs_folder, ...
                                      screen_pattern_width_mm, ...
                                      Np, show_pattern_order, 'SCREEN');
    save(partA_savefile, ...
         'A_screen','A_screen_noref','H_screen','H_screen_noref', ...
         'R_screen','T_screen','img_sizes_screen','img_files_screen');
end

%% ===================================================================
%  PART B: PHYSICAL PATTERN CALIBRATION
%% ===================================================================
fprintf('\n============================================================\n');
fprintf('  PART B: Camera calibration with PHYSICAL pattern\n');
fprintf('============================================================\n');

[A_phys, A_phys_noref, H_phys, H_phys_noref, R_phys, T_phys, ...
 img_sizes_phys, img_files_phys] = ...
        calibrate_camera_section1(phys_imgs_folder, ...
                                  phys_pattern_width_mm, ...
                                  Np, show_pattern_order, 'PHYSICAL');

save('Section1_PartB_physical.mat', ...
     'A_phys','A_phys_noref','H_phys','H_phys_noref', ...
     'R_phys','T_phys','img_sizes_phys','img_files_phys');

%% ===================================================================
%  FINAL COMPARISON BETWEEN A (screen) and A' (physical)
%% ===================================================================
fprintf('\n============================================================\n');
fprintf('  FINAL COMPARISON  A  vs  A''   \n');
fprintf('============================================================\n');

fprintf('\n--- Intrinsic matrix from SCREEN pattern (A) ---\n');
disp(A_screen);

fprintf('\n--- Intrinsic matrix from PHYSICAL pattern (A'') ---\n');
disp(A_phys);

rel_diff = 100 * abs(A_screen - A_phys) ./ max(abs(A_screen), 1e-9);
fprintf('\n--- Element-wise relative difference  |A-A''|/|A|  (in %%) ---\n');
disp(rel_diff);

fprintf('\nSection 1 done.  Results saved in Section1_PartA_screen.mat and Section1_PartB_physical.mat\n');


%% ===================================================================
%  FUNCTION: full calibration pipeline for one set of images
%% ===================================================================
function [A_refined, A_noref, H_refined, H_noref, R, T, img_sizes, img_files] = ...
            calibrate_camera_section1(folder, pattern_width_mm, Np, show_order, tag)

    img_files = [dir(fullfile(folder,'*.jpg'));
                 dir(fullfile(folder,'*.jpeg'));
                 dir(fullfile(folder,'*.png'))];
    n_imgs = numel(img_files);
    if n_imgs < 3
        error('Need at least 3 images in %s (found %d).', folder, n_imgs);
    end
    fprintf('Found %d images in %s\n', n_imgs, folder);

    [real_pts_2xN, pattern_img] = get_real_points_checkerboard_vmmc( ...
                                       Np, pattern_width_mm, show_order);
    real_pts = real_pts_2xN';

    H_noref   = cell(1, n_imgs);
    H_refined = cell(1, n_imgs);
    img_sizes = zeros(n_imgs, 2);

    for k = 1:n_imgs
        img_path = fullfile(folder, img_files(k).name);
        I = imread(img_path);
        img_sizes(k, :) = [size(I,2), size(I,1)];

        fprintf('\n[%s] Image %d/%d: %s\n', tag, k, n_imgs, img_files(k).name);
        fprintf('  -> Click the %d pattern points in the SAME ORDER\n', Np);
        fprintf('     shown on the synthetic pattern figure.\n');
        fprintf('     Use LEFT click for the first %d, RIGHT click for the LAST one.\n', Np-1);

        clicked = get_user_points_vmmc(I);
        if size(clicked, 2) ~= Np
            error('Wrong number of points clicked in image %d (got %d, expected %d)', ...
                   k, size(clicked,2), Np);
        end

        Hk = homography_solve_vmmc(real_pts, clicked);
        H_noref{k} = Hk;

        [Hk_ref, rperr] = homography_refine_vmmc(real_pts, clicked, Hk);
        H_refined{k} = Hk_ref;
        fprintf('  -> Homography refined, mean reprojection error = %.4f\n', rperr);
    end

    A_refined = internal_parameters_solve_vmmc(H_refined);
    A_noref   = internal_parameters_solve_vmmc(H_noref);

    [R, T] = external_parameters_solve_vmmc(A_refined, H_refined);

    fprintf('\n--- [%s] A (refined homographies) ---\n', tag);
    disp(A_refined);
    fprintf('--- [%s] A (NON-refined homographies) ---\n', tag);
    disp(A_noref);

    alpha  = A_refined(1,1);
    beta   = A_refined(2,2);
    gamma  = A_refined(1,2);
    u0     = A_refined(1,3);
    v0     = A_refined(2,3);
    mean_w = mean(img_sizes(:,1));
    mean_h = mean(img_sizes(:,2));

    % --- Angle between the image axes -----------------------------------
    % In A, gamma = -alpha * cot(theta), so theta = acot(-gamma/alpha).
    % acotd returns values in (-90, 90); the geometric angle between two
    % axes lies in (0, 180), with 90 meaning perpendicular. Wrap negative
    % outputs by adding 180 so we always report the correct theta.
    if gamma == 0
        angle_axes_deg = 90;
    else
        angle_axes_deg = acotd(-gamma / alpha);
        if angle_axes_deg < 0
            angle_axes_deg = angle_axes_deg + 180;
        end
    end

    fprintf('\n   ---  Quick analysis (%s) ---\n', tag);
    fprintf('   alpha (a)      = %.3f\n', alpha);
    fprintf('   beta  (b)      = %.3f\n', beta);
    fprintf('   gamma (skew)   = %.6f\n', gamma);
    fprintf('   u0 (princ. pt) = %.3f   vs  image-centre u = %.1f\n', u0, mean_w/2);
    fprintf('   v0 (princ. pt) = %.3f   vs  image-centre v = %.1f\n', v0, mean_h/2);
    fprintf('   |a-b|/a        = %.2f%%  (pixel squareness)\n', 100*abs(alpha-beta)/alpha);
    fprintf('   Angle between image axes = %.4f deg (90 deg => perfectly orthogonal)\n', ...
            angle_axes_deg);
    fprintf('   Deviation from 90 deg    = %.4f deg\n', abs(angle_axes_deg - 90));
end