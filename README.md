# Multi-view 3D reconstruction from a camera (using MATLAB)

*3D Vision for Multiple and Moving Cameras* (M.Sc. IPCVai). 

Calibrated a Samsung Galaxy S24 camera, matched features across nine photos of a small scene, and reconstructed the scene in 3D: first in a projective frame, then upgraded to a metric (Euclidean) one. The full result analysis and experiments with all figures and tables are in [`Report/`](Report/).

## Results

| Step | Result |
|---|---|
| Camera calibration (screen checkerboard, 9 views) | Mean reprojection error 1.08–1.91 px per view; pixels square to 0.024 %, image axes within 0.25° of orthogonal |
| Calibration with a hand-drawn 3×3 pattern (6 views) | Focal length up to 14 % off the checkerboard result, which shows how much point count and image coverage matter |
| SIFT + RANSAC matching, best pair (views 2–7) | 4,104 of 4,851 matches kept as inliers (84.6 %) |
| N-view tracks | 7,498 tracks seen in at least 3 views (3.8 views on average) |
| Final metric reconstruction | 4 cameras, 967 points, median reprojection error 1.95 px (mean 3.44 px) |

## How it works

1. **Calibration.** Zhang's method with a checkerboard displayed on a laptop screen (8×8 inner corners, 25 mm spacing). I repeated it with a hand-drawn dot grid on graph paper and compared the two intrinsic matrices.
2. **Pairwise matching.** MATLAB SIFT, Lowe ratio test (0.7), then RANSAC fits of the fundamental matrix and of a homography on seven image pairs, from very short to very wide baselines. VLFeat SIFT gave almost the same numbers and served as a cross-check. I also matched one SIFT octave at a time to see which scales give the most reliable matches.
3. **Tracks.** All 36 image pairs were matched and filtered with RANSAC, and the matches were linked into tracks with union-find. A track is kept if it appears in 3 or more views with at most one keypoint per image.
4. **Projective reconstruction.** Initial pair 2–7 from F, linear triangulation, resectioning of the other cameras (normalised DLT), then projective bundle adjustment.
5. **Euclidean upgrade.** Essential matrix from F and K, the (R, t) solution chosen by cheirality (all points in front of both cameras), then Euclidean resectioning of the remaining cameras. Cameras with a median error above 5 px were rejected.
6. **Point cloud**, coloured with pixel values from image 2.

## What went wrong, and what I changed

The scene turned out to be almost flat: thin objects on a table, with only 1–3 cm of height. This made projective bundle adjustment badly conditioned (near-singular Jacobian), and it stopped without improving the error. I ran the reconstruction three times:

| Version | Change | Outcome |
|---|---|---|
| v1 | All 9 cameras, no outlier handling | BA failed, final mean error 355 px |
| v2 | Weak cameras dropped, observations pruned before BA | BA converged, but only 45 points were left |
| v3 (final) | Weak cameras dropped, no pruning before BA, cameras rejected after the Euclidean upgrade | 4 cameras, 967 points, median 1.95 px |

On a near-planar scene, pruning before BA throws away good observations, because the projective frame is distorted for cameras far from the initial pair. Rejecting outliers after the metric upgrade worked better. Next time I would add height to the scene (for example a book under one object) and use a bundle adjustment with explicit gauge fixing.

## Data

All photos were taken with the rear 1× lens of a Galaxy S24 (4000×3000 px, fixed zoom and focus, no portrait mode), so one set of intrinsics applies to every image. The full-resolution originals are available on the Releases page.
