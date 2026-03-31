# Instance Method Comparison

## Inputs
- `cc`: /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/eval_building_seg_object_cc, /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/analyze_building_seg_object_cc (artifacts: bucket_summary, instance_metrics_summary, summary)
- `watershed`: /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/eval_building_seg_object_watershed, /home/ubuntu/code/lky/code/refactor/xbd_building_eval/outputs/analyze_building_seg_object_watershed (artifacts: bucket_summary, instance_metrics_summary, summary)

## Protocol Guardrails
- GT fixation check: no. Observed `gt_method` values: cc, watershed.
- Bucket-reference GT fixation: no. Observed `bucket_reference_gt_method` values: cc, watershed.
- `gt_instance_count` consistency: no, counts differ across runs (`cc`=35822, `watershed`=38686).

## Overall
- Best `object_f1`: `cc` at 0.6896.

## Bucket Recall
- `dense` bucket winner: `cc` with recall 0.6629.
- `medium` bucket winner: `cc` with recall 0.4200.
- `small` bucket winner: `cc` with recall 0.0594.
- `merge_count` vs baseline `cc`: `watershed` increases it by 420.
- `split_count` vs baseline `cc`: no increase (1657 -> 1511).
