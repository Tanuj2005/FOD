# EDA report: ds1_runway_fod_detector

## Headline numbers

| metric | value |
|---|---|
| dataset | ds1_runway_fod_detector |
| root | Runway FOD Detector.v1i.yolov11 |
| classes | 2 |
| train_images | 9054 |
| train_instances | 8861 |
| train_background_pct | 2.44 |
| valid_images | 881 |
| valid_instances | 868 |
| valid_background_pct | 1.59 |
| test_images | 417 |
| test_instances | 416 |
| test_background_pct | 0.96 |
| total_images | 10352 |
| total_instances | 10145 |
| boxes_per_image_mean | 0.98 |
| boxes_per_image_max | 8 |
| unique_resolutions | 1 |
| most_common_resolution | 640x640 |
| image_modes | RGB |
| polygon_labels_converted | 36 |
| issue_corrupt_image | 0 |
| issue_missing_label_file | 0 |
| issue_malformed_line | 0 |
| issue_class_id_out_of_range | 0 |
| issue_non_positive_size | 0 |
| issue_out_of_bounds | 0 |
| issue_duplicate_box | 0 |
| issue_unreadable_label | 0 |
| orphan_label_files | 0 |
| imbalance_ratio_max_over_min | 306.4 |
| rare_classes_lt_threshold | Runway-FOD-Detector |
| n_rare_classes | 1 |
| classes_missing_in_valid | Runway-FOD-Detector |
| classes_missing_in_test | Runway-FOD-Detector |
| pct_small_objects | 0.7 |
| pct_medium_objects | 56.5 |
| pct_large_objects | 42.7 |
| median_box_area_pct_of_image | 1.8461 |
| median_box_min_side_px | 71.0 |
| pct_boxes_min_side_lt_8px | 0.1 |
| pct_boxes_min_side_lt_16px | 0.4 |
| median_aspect_ratio_w_over_h | 0.94 |
| brightness_mean | 146.7 |
| brightness_std_across_images | 19.4 |
| pct_dark_images_lt_60 | 0.0 |
| pct_bright_images_gt_190 | 1.3 |
| pct_grayscale_images | 52.0 |
| sharpness_median_lapvar | 884.4 |
| pct_blurry_lapvar_lt_30 | 0.0 |
| valid_near_dup_of_train | 673 |
| valid_near_dup_of_train_pct | 76.39 |
| valid_exact_dup_of_train | 384 |
| valid_median_dist_to_train | 5.0 |
| test_near_dup_of_train | 326 |
| test_near_dup_of_train_pct | 78.18 |
| test_exact_dup_of_train | 193 |
| test_median_dist_to_train | 5.0 |
| test_near_dup_of_valid | 150 |
| train_images_with_near_dup_in_train | 2491 |
| train_images_with_near_dup_in_train_pct | 27.51 |
| unique_base_names | 1485 |
| base_names_with_multiple_files | 1465 |
| files_that_are_augmented_copies | 8867 |
| base_names_spanning_multiple_splits | 899 |

## Per-class table

| class_id | class_name | total | train | valid | test | pct_small | median_area_px | median_min_side_px | median_aspect |
|---|---|---|---|---|---|---|---|---|---|
| 0 | FOD | 10112 | 8828 | 868 | 416 | 0.5 | 7584.5 | 71.0 | 0.94 |
| 1 | Runway-FOD-Detector | 33 | 33 | 0 | 0 | 54.5 | 416.2 | 16.7 | 1.33 |

## Suggested cleaning / design actions (rule-based, review them)

- 8867 files look like augmented copies: dedupe by base_name or re-download without augmentation.
- Same source image appears in multiple splits (leakage): re-split by base_name.
- 76.39% of valid images are near-duplicates of train: re-split by scene/sequence or remove them from valid.
- 78.18% of test images are near-duplicates of train: re-split by scene/sequence or remove them from test.
- Many near-duplicate images inside train (likely video frames): consider thinning them.
- Train background-only images = 2.44%: add hard negatives up to ~5-10%.
- Rare classes (<50 instances): Runway-FOD-Detector. Merge, oversample, or document.
- Classes absent from test: Runway-FOD-Detector. Stratify the split.
- Classes absent from valid: Runway-FOD-Detector. Stratify the split.
- 52.0% of images are grayscale: colour cues unavailable.
- Almost no dark/very bright images: night/low-light generalisation cannot be evaluated with this data alone.
