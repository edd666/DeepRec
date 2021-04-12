# deeprec

根据推荐系统的不同阶段以及优化目标，可分为：

- 召回（Recall）
- 排序（Ranking）
  - 点击率（CTR）
  - 转化率（CVR）
  - 多目标（Multi task learning）



**注意：**

- 对于序列特征，仅支持mask_zero格式--不足长度的补0，不支持直接指定序列长度。
- 