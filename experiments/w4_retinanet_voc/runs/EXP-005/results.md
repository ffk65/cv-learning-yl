# EXP-005 评估结果（val）

- 权重：`checkpoints/last.pth`
- 指标口径：VOC mAP@0.5（与 mmdet `mAP@0.5` 一致）
- **mAP@0.5 = 37.22%**

| 类别 | AP(%) | 类别 | AP(%) |
| --- | --- | --- | --- |
| car | 61.0 | dog | 36.3 |
| person | 58.3 | cow | 34.2 |
| aeroplane | 56.6 | bus | 32.4 |
| bicycle | 56.4 | chair | 25.1 |
| horse | 53.1 | sofa | 25.0 |
| train | 45.2 | boat | 22.9 |
| cat | 42.8 | bird | 22.7 |
| tvmonitor | 42.8 | sheep | 22.3 |
| motorbike | 42.8 | bottle | 16.3 |
| diningtable | 39.8 | pottedplant | 8.5 |

## 复现这条结果

```powershell
python -m trainer.eval --ckpt checkpoints/last.pth --exp EXP-005
```
