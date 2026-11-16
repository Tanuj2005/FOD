# EDA report: ds2_fod_runway

## Headline numbers

| metric | value |
|---|---|
| dataset | ds2_fod_runway |
| root | FOD Runway Dataset.v2i.yolov11 |
| classes | 1 |
| train_images | 5004 |
| train_instances | 5002 |
| train_background_pct | 0.08 |
| valid_images | 641 |
| valid_instances | 643 |
| valid_background_pct | 0.0 |
| test_images | 338 |
| test_instances | 338 |
| test_background_pct | 0.0 |
| total_images | 5983 |
| total_instances | 5983 |
| boxes_per_image_mean | 1.0 |
| boxes_per_image_max | 2 |
| unique_resolutions | 1 |
| most_common_resolution | 640x640 |
| image_modes | RGB |
| polygon_labels_converted | 0 |
| issue_corrupt_image | 0 |
| issue_missing_label_file | 0 |
| issue_malformed_line | 0 |
| issue_class_id_out_of_range | 0 |
| issue_non_positive_size | 0 |
| issue_out_of_bounds | 0 |
| issue_duplicate_box | 0 |
| issue_unreadable_label | 0 |
| orphan_label_files | 0 |
| imbalance_ratio_max_over_min | 1.0 |
| rare_classes_lt_threshold |  |
| n_rare_classes | 0 |
| classes_missing_in_valid |  |
| classes_missing_in_test |  |
| pct_small_objects | 28.8 |
| pct_medium_objects | 26.8 |
| pct_large_objects | 44.4 |
| median_box_area_pct_of_image | 1.6575 |
| median_box_min_side_px | 65.5 |
| pct_boxes_min_side_lt_8px | 0.1 |
| pct_boxes_min_side_lt_16px | 9.2 |
| median_aspect_ratio_w_over_h | 1.18 |
| brightness_mean | 169.7 |
| brightness_std_across_images | 42.9 |
| pct_dark_images_lt_60 | 0.0 |
| pct_bright_images_gt_190 | 28.7 |
| pct_grayscale_images | 23.7 |
| sharpness_median_lapvar | 922.6 |
| pct_blurry_lapvar_lt_30 | 0.0 |
| valid_near_dup_of_train | 77 |
| valid_near_dup_of_train_pct | 12.01 |
| valid_exact_dup_of_train | 5 |
| valid_median_dist_to_train | 20.0 |
| test_near_dup_of_train | 55 |
| test_near_dup_of_train_pct | 16.27 |
| test_exact_dup_of_train | 5 |
| test_median_dist_to_train | 19.0 |
| test_near_dup_of_valid | 33 |
| train_images_with_near_dup_in_train | 1143 |
| train_images_with_near_dup_in_train_pct | 22.84 |
| unique_base_names | 2420 |
| base_names_with_multiple_files | 1762 |
| files_that_are_augmented_copies | 3563 |
| base_names_spanning_multiple_splits | 168 |

## Per-class table

| class_id | class_name | total | train | valid | test | pct_small | median_area_px | median_min_side_px | median_aspect |
|---|---|---|---|---|---|---|---|---|---|
| 0 | FOD | 5983 | 5002 | 643 | 338 | 28.8 | 6789.0 | 65.5 | 1.18 |

## Suggested cleaning / design actions (rule-based, review them)

- 3563 files look like augmented copies: dedupe by base_name or re-download without augmentation.
- Same source image appears in multiple splits (leakage): re-split by base_name.
- 12.01% of valid images are near-duplicates of train: re-split by scene/sequence or remove them from valid.
- 16.27% of test images are near-duplicates of train: re-split by scene/sequence or remove them from test.
- Many near-duplicate images inside train (likely video frames): consider thinning them.
- Train background-only images = 0.08%: add hard negatives up to ~5-10%.
- 23.7% of images are grayscale: colour cues unavailable.
