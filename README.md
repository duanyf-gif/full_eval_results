# Full Eval Results

本仓库用于存放 EmoArt 验证集上的完整生成结果、评测输出和验证集图片。

## 目录说明

- `emoart_val/`
  - 本次评测使用的 EmoArt 验证集划分，也就是真实图片。
  - 包含 `annotation.json` 和按风格/类别组织的验证集图片。
- `show-o/baseline/`
  - Show-O baseline 评测结果。
  - 包含 `t2i_results.json`、`mmu_results.json`、`eval_config.json`、日志文件和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `show-o/itfi_layer23_top50_lam2p0/`
  - 使用 ITFI 神经元干预方法后的 Show-O T2I 评测结果。
  - `t2i_images/` 中包含 1660 张生成图片。
- `janus/emoart_janus_baseline_full_val/`
  - Janus-Pro-1B baseline 在 EmoArt full validation split 上的 T2I 评测结果。
  - 包含 `t2i_results.json`、`eval_config.json`、运行日志和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `janus/emoart_janus_itfi/`
  - Janus-Pro-1B 使用 ITFI 后的 T2I 评测结果。
  - 包含 `t2i_results.json`、`eval_config.json`、运行日志和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。

## 目录结构

```text
.
|-- emoart_val/
|   |-- annotation.json
|   `-- Images/
|-- janus/
|   |-- emoart_janus_baseline_full_val/
|   |   |-- eval_config.json
|   |   |-- full_val_background.log
|   |   |-- t2i_images/
|   |   `-- t2i_results.json
|   `-- emoart_janus_itfi/
|       |-- eval_config.json
|       |-- run_1600_score.log
|       |-- t2i_images/
|       `-- t2i_results.json
`-- show-o/
    |-- baseline/
    |   |-- eval_config.json
    |   |-- logs/
    |   |-- mmu_results.json
    |   |-- t2i_images/
    |   `-- t2i_results.json
    `-- itfi_layer23_top50_lam2p0/
        |-- eval_config.json
        |-- t2i_images/
        `-- t2i_results.json
```
