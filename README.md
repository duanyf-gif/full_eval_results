# Full Eval Results

本仓库用于存放 EmoArt 和 Artemis 数据集上的完整生成结果、评测输出和验证集图片。

## 目录说明

- `emoart_val/`
  - 本次评测使用的 EmoArt 验证集划分，也就是真实图片。
  - 包含 `annotation.json` 和按风格/类别组织的验证集图片。
- `Artemis/`
  - 本次评测使用的 Artemis 测试集，也就是真实图片。
  - 包含 `annotation.json` 和按风格组织的测试集图片。
  - `Images/` 中包含 1487 张图片，覆盖 27 个风格目录。
- `FLUX.1-dev/outputs/artemis_test_flux_dev_full/`
  - FLUX.1-dev 在 Artemis 数据集上的生图结果。
  - 包含 `t2i_results.json`、`t2i_results.partial.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。
- `FLUX.1-dev/outputs/emoart_val_flux_dev_full/`
  - FLUX.1-dev 在 EmoArt full validation split 上的生图结果。
  - 包含 `generation_config.json`、`t2i_results.json`、`t2i_results.partial.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `SDv1.5/outputs/artemis_test_sd15/`
  - Stable Diffusion v1.5 在 Artemis 数据集上的生图结果。
  - 包含 `generation_config.json`、`t2i_results.json`、`t2i_results.partial.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。
- `SDv1.5/outputs/emoart_val_sd15/`
  - Stable Diffusion v1.5 在 EmoArt full validation split 上的生图结果。
  - 包含 `generation_config.json`、`t2i_results.json`、`t2i_results.partial.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `show-o/baseline/`
  - Show-O baseline 评测结果。
  - 包含 `t2i_results.json`、`mmu_results.json`、`eval_config.json`、日志文件和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `show-o/itfi_layer23_top50_lam2p0/`
  - 使用 ITFI 神经元干预方法后的 Show-O T2I 评测结果。
  - `t2i_images/` 中包含 1660 张生成图片。
- `show-o/ola_test_baseline/`
  - Show-O baseline 在 Artemis 数据集上的生图结果。
  - 包含 `eval_config.json`、`t2i_results.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。
- `show-o/ola_test_itfi_layer23_top50_lam2p0/`
  - Show-O ITFI layer23 top50 lambda=2.0 在 Artemis 数据集上的生图结果。
  - 包含 `eval_config.json`、`t2i_results.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。
- `janus/emoart_janus_baseline_full_val/`
  - Janus-Pro-1B baseline 在 EmoArt full validation split 上的 T2I 评测结果。
  - 包含 `t2i_results.json`、`eval_config.json`、运行日志和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `janus/emoart_janus_itfi/`
  - Janus-Pro-1B 使用 ITFI 后的 T2I 评测结果。
  - 包含 `t2i_results.json`、`eval_config.json`、运行日志和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `janus/topK100_lam1p2/`
  - Janus-Pro-1B layer23 TopK=100、lambda=1.2 的 auto-tune 评测结果。
  - 包含 `eval_config.json`、`t2i_results.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1660 张生成图片。
- `janus/ola_test_janus_baseline_full/`
  - Janus-Pro-1B baseline 在 Artemis 数据集上的生图结果。
  - 包含 `eval_config.json`、`t2i_results.json`、`run.log`、`scores/` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。
- `janus/ola_test_janus_itfi_layer23_top100_lam1p2_full/`
  - Janus-Pro-1B ITFI layer23 TopK=100、lambda=1.2 在 Artemis 数据集上的生图结果。
  - 包含 `eval_config.json`、`t2i_results.json`、`run.log` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。
- `janus/ola_test_janus_itfi_layer21_22_23_top100_lam1p0_full/`
  - Janus-Pro-1B ITFI layer21/22/23 TopK=100、lambda=1.0 在 Artemis 数据集上的生图结果。
  - 包含 `eval_config.json`、`t2i_results.json`、`run.log` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。
- `janus/ola_test_janus_itfi_layer22_23_top100_lam1p1_full/`
  - Janus-Pro-1B ITFI layer22/23 TopK=100、lambda=1.1 在 Artemis 数据集上的生图结果。
  - 包含 `eval_config.json`、`t2i_results.json` 和生成的 T2I 图片。
  - `t2i_images/` 中包含 1487 张生成图片。

## 目录结构

```text
.
|-- Artemis/
|   |-- annotation.json
|   `-- Images/
|-- FLUX.1-dev/
|   `-- outputs/
|       |-- artemis_test_flux_dev_full/
|       |   |-- t2i_images/
|       |   |-- t2i_results.json
|       |   `-- t2i_results.partial.json
|       `-- emoart_val_flux_dev_full/
|           |-- generation_config.json
|           |-- t2i_images/
|           |-- t2i_results.json
|           `-- t2i_results.partial.json
|-- SDv1.5/
|   `-- outputs/
|       |-- artemis_test_sd15/
|       |   |-- generation_config.json
|       |   |-- t2i_images/
|       |   |-- t2i_results.json
|       |   `-- t2i_results.partial.json
|       `-- emoart_val_sd15/
|           |-- generation_config.json
|           |-- t2i_images/
|           |-- t2i_results.json
|           `-- t2i_results.partial.json
|-- emoart_val/
|   |-- annotation.json
|   `-- Images/
|-- janus/
|   |-- emoart_janus_baseline_full_val/
|   |   |-- eval_config.json
|   |   |-- full_val_background.log
|   |   |-- t2i_images/
|   |   `-- t2i_results.json
|   |-- emoart_janus_itfi/
|   |   |-- eval_config.json
|   |   |-- run_1600_score.log
|   |   |-- t2i_images/
|   |   `-- t2i_results.json
|   |-- ola_test_janus_baseline_full/
|   |   |-- eval_config.json
|   |   |-- run.log
|   |   |-- scores/
|   |   |-- t2i_images/
|   |   `-- t2i_results.json
|   |-- ola_test_janus_itfi_layer23_top100_lam1p2_full/
|   |   |-- eval_config.json
|   |   |-- run.log
|   |   |-- t2i_images/
|   |   `-- t2i_results.json
|   |-- ola_test_janus_itfi_layer21_22_23_top100_lam1p0_full/
|   |   |-- eval_config.json
|   |   |-- run.log
|   |   |-- t2i_images/
|   |   `-- t2i_results.json
|   |-- ola_test_janus_itfi_layer22_23_top100_lam1p1_full/
|   |   |-- eval_config.json
|   |   |-- t2i_images/
|   |   `-- t2i_results.json
|   `-- topK100_lam1p2/
|       |-- eval_config.json
|       |-- t2i_images/
|       `-- t2i_results.json
`-- show-o/
    |-- baseline/
    |   |-- eval_config.json
    |   |-- logs/
    |   |-- mmu_results.json
    |   |-- t2i_images/
    |   `-- t2i_results.json
    |-- itfi_layer23_top50_lam2p0/
    |   |-- eval_config.json
    |   |-- t2i_images/
    |   `-- t2i_results.json
    |-- ola_test_baseline/
    |   |-- eval_config.json
    |   |-- t2i_images/
    |   `-- t2i_results.json
    `-- ola_test_itfi_layer23_top50_lam2p0/
        |-- eval_config.json
        |-- t2i_images/
        `-- t2i_results.json
```
