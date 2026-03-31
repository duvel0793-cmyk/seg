# Merge Candidate Report

## Top Candidate Sources
- Candidate pool above score threshold: 7064
- Exported top subset: 200
- Top-20 candidates come from 16 unique images
| image_id | top20_count | best_rank | best_score | density_bucket | merge_count | fn |
| --- | --- | --- | --- | --- | --- | --- |
| mexico-earthquake_00000094 | 3 | 2 | 11.0 | dense | 22 | 65 |
| mexico-earthquake_00000139 | 2 | 1 | 12.0 | dense | 23 | 180 |
| palu-tsunami_00000081 | 2 | 15 | 10.0 | dense | 19 | 73 |
| palu-tsunami_00000082 | 1 | 3 | 11.0 | dense | 22 | 58 |
| palu-tsunami_00000120 | 1 | 5 | 11.0 | dense | 24 | 64 |
| mexico-earthquake_00000005 | 1 | 6 | 11.0 | dense | 23 | 74 |
| mexico-earthquake_00000117 | 1 | 7 | 11.0 | dense | 69 | 344 |
| palu-tsunami_00000181 | 1 | 8 | 11.0 | dense | 32 | 79 |
| palu-tsunami_00000165 | 1 | 9 | 11.0 | dense | 9 | 28 |
| hurricane-matthew_00000108 | 1 | 10 | 11.0 | dense | 7 | 27 |

## Candidate Distribution
- By image density bucket: dense=6948, medium=115, sparse=1
- By local density bucket: dense=5777, medium=837, sparse=450

## Hardest Failure Overlap
- Hardest failure image pool from `top_failure_cases`: 66 unique images
- Exported candidate subset overlaps 31 of them
- Overlapping image ids: hurricane-harvey_00000456, hurricane-harvey_00000479, mexico-earthquake_00000005, mexico-earthquake_00000006, mexico-earthquake_00000010, mexico-earthquake_00000058, mexico-earthquake_00000061, mexico-earthquake_00000063, mexico-earthquake_00000069, mexico-earthquake_00000076, mexico-earthquake_00000080, mexico-earthquake_00000117, mexico-earthquake_00000121, mexico-earthquake_00000124, mexico-earthquake_00000131, mexico-earthquake_00000135, mexico-earthquake_00000139, mexico-earthquake_00000161, mexico-earthquake_00000163, mexico-earthquake_00000180

## Top-100 / Top-200 Coverage
- Merge-heavy image definition: top 34 merge-positive images ranked by `merge_count`
- Top 100 candidates cover 22 merge-heavy images, coverage=0.647059
- Top 200 candidates cover 25 merge-heavy images, coverage=0.735294
- Total `merged_into_other` GT samples: 5342
- Top 100 candidates hit 681 of them, coverage=0.12748
- Top 200 candidates hit 1071 of them, coverage=0.200487

## Automatic Summary
- 候选高度集中于 dense hardest blobs，当前排序已经明显把高压 merge 口袋提到了前面。
- 这批候选已经能覆盖相当一部分 merge-heavy 图像或 `merged_into_other` 样本，值得进入下一步 local refine / SAM3 feasibility 分支。
