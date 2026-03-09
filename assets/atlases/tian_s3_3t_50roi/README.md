# Tian Subcortex 50 ROI

这个目录存放 Tian Scale III 3T 亚皮层 atlas 的 50 ROI 版本。

文件说明：

- `Tian_Subcortex_S3_3T_50ROI.nii.gz`
- `Tian_Subcortex_S3_3T_50ROI_labels.txt`

说明：

- 这个 NIfTI 本身就只有 50 个非零标签，标签编号为 `1..50`。
- 因此“提取前 50 个 ROI”在这个 atlas 上等价于保留整个 atlas。
- 这 50 个 ROI 对应标签文本文件中的 50 行名称。

当前建议用途：

- 如果只想单独使用 Tian 的 50 个亚皮层 ROI，可以把这个 NIfTI 直接传给预处理脚本的 `--atlas-labels-img`。
- 如果要构造 450 ROI 版本，需要再和 400 ROI 的 Schaefer cortical atlas 做标签级合并，不能只靠复制文件名完成。

示例路径：

- `assets/atlases/tian_s3_3t_50roi/Tian_Subcortex_S3_3T_50ROI.nii.gz`