# EXP-002 评估结果（val）

- 权重：`checkpoints/last.pth`
- 指标口径：VOC mAP@0.5（与 mmdet `mAP@0.5` 一致）
- **mAP@0.5 = 33.34%**

| 类别 | AP(%) | 类别 | AP(%) |
| --- | --- | --- | --- |
| person | 55.4 | dog | 35.2 |
| car | 54.5 | bus | 34.6 |
| cat | 52.8 | sofa | 24.4 |
| horse | 48.9 | chair | 22.9 |
| train | 48.4 | cow | 17.3 |
| bicycle | 47.5 | boat | 16.7 |
| aeroplane | 47.3 | bird | 13.6 |
| motorbike | 42.6 | sheep | 11.1 |
| diningtable | 40.7 | bottle | 10.4 |
| tvmonitor | 39.8 | pottedplant | 2.5 |

## 复现这条结果

```powershell
python -m trainer.eval --ckpt checkpoints/last.pth --exp EXP-002
```
