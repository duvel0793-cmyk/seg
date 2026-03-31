# Instance Method Comparison

## Inputs
- `cc`: /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/eval_building_seg_object_cc, /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/analyze_building_seg_object_cc (artifacts: bucket_summary, instance_metrics_summary, summary)
- `watershed`: /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/eval_building_seg_object_watershed, /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/analyze_building_seg_object_watershed (artifacts: bucket_summary, instance_metrics_summary, summary)

## Protocol Guardrails
- GT fixation check: yes
- Observed `gt_method` values: cc
- Bucket reference fixation check: yes
- Observed `bucket_reference_gt_method` values: cc
- `gt_instance_count` identical across runs: yes (35822)

## Overall
- Higher `object_f1` pred_method: `watershed` at 0.6948.
- Lower `merge_count` pred_method: `watershed` at 2416.0000.
- Lower `split_count` pred_method: `cc` at 1657.0000.

## Bucket Recall
- `dense` bucket winner: tied at 0.6784 across `cc`, `watershed`.
- `medium` bucket winner: tied at 0.4313 across `cc`, `watershed`.
- `small` bucket winner: tied at 0.0625 across `cc`, `watershed`.
