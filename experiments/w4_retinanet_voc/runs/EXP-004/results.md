# EXP-004 评估结果（val）

- 权重：`checkpoints/last.pth`
- 指标口径：VOC mAP@0.5（与 mmdet `mAP@0.5` 一致）
- **mAP@0.5 = 8.38%**

| 类别 | AP(%) | 类别 | AP(%) |
| --- | --- | --- | --- |
| person | 46.0 | bus | 4.7 |
| car | 36.7 | chair | 4.3 |
| dog | 11.7 | sofa | 3.7 |
| horse | 11.0 | aeroplane | 3.2 |
| cow | 7.7 | train | 2.9 |
| motorbike | 7.3 | bottle | 2.2 |
| diningtable | 5.5 | boat | 1.6 |
| tvmonitor | 5.5 | bird | 1.0 |
| cat | 5.5 | sheep | 0.9 |
| bicycle | 5.4 | pottedplant | 0.7 |

## 复现这条结果

```powershell
python -m trainer.eval --ckpt checkpoints/last.pth --exp EXP-004
```
