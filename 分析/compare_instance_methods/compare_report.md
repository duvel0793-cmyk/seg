# Instance Method Comparison

## Inputs
- `cc`: /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/eval_building_seg_object_cc, /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/analyze_building_seg_object_cc (artifacts: bucket_summary, instance_metrics_summary, summary)
- `watershed`: /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/eval_building_seg_object_watershed, /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/analyze_building_seg_object_watershed (artifacts: bucket_summary, instance_metrics_summary, summary)

## Overall
- Best `object_f1`: `cc` (0.6896), vs `watershed` (0.6773), delta 0.0123.
- Lowest `merge_count`: `cc` (2649), vs `watershed` (3069), lower by 420.
- `split_count`: `cc` is highest at 1657 vs `watershed` at 1511 (+146, +9.7%), no notable increase.

## Bucket Recall
- `gt_area_bucket/small` recall: `cc` (0.0594) vs `watershed` (0.0524), delta 0.0070.
- `gt_area_bucket/medium` recall: `cc` (0.4200) vs `watershed` (0.3896), delta 0.0304.
- `image_density_bucket/dense` recall: `cc` (0.6629) vs `watershed` (0.6326), delta 0.0303.
