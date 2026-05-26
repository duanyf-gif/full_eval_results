# Score Tools

这个目录用于批量调用 OpenAI 兼容接口，对生成图片做自动评分，并把结果保存成 `json/csv` 文件。

## 文件说明

- `score_images.py`
  - 对每张图片输出 5 个维度评分：`brushstroke`、`color`、`composition`、`light_and_shadow`、`line_quality`。
- `score_total_images.py`
  - 对每张图片输出 1 个整体质量分：`overall_quality_score`，范围是 1 到 10。
- `convert_scores_to_numeric.py`
  - 把 `score_images.py` 生成的文本等级分数转换成数值分数，并在 CSV 末尾追加平均值。
- `prompt.txt`
  - 五维评分脚本默认使用的提示词。
- `total_eva_prompt`
  - 整体评分脚本默认使用的提示词。

## 环境准备

需要先安装 Python 依赖：

```bash
pip install openai
```

需要准备 OpenAI 兼容接口的认证信息。脚本会使用标准环境变量：

```bash
export OPENAI_API_KEY=your_api_key
```

如果使用自定义兼容接口，也可以设置：

```bash
export OPENAI_BASE_URL=https://your-base-url/v1/
```

脚本内部默认会优先读取命令行 `--base_url`，其次读取 `OPENAI_BASE_URL`。

## 输出位置

运行评分脚本后，会自动在 `score/` 下创建：

- `results/`
  - 保存评分结果的 `json` 和 `csv`。
- `logs/`
  - 保存运行日志和报错信息。

## 用法

五维评分：

```bash
python score/score_images.py \
  --images_dir ../show-o/baseline/t2i_images \
  --score_dir . \
  --output_prefix results_showo_baseline
```

常用参数：

- `--images_dir`：要评分的图片目录。
- `--score_dir`：结果和日志输出目录。这里建议传 `score` 目录本身，或在 `score/` 目录内运行时传 `.`。
- `--output_prefix`：输出文件名前缀。
- `--max_images`：只处理前几张图片，适合测试。
- `--model`：评分模型名，默认是 `gpt-5.4`。
- `--base_url`：自定义 OpenAI 兼容接口地址。
- `--timeout`：单次请求超时秒数。
- `--sleep_seconds`：每次请求后的等待时间。
- `--image_detail`：图片细节级别，默认是"high"
整体质量评分：

```bash
python score/score_total_images.py \
  --images_dir ../show-o/baseline/t2i_images \
  --prompt_path total_eva_prompt \
  --score_dir . \
  --output_prefix results_showo_baseline_total
```

和五维评分相比，额外参数是：

- `--prompt_path`：整体评分提示词文件路径，默认使用 `total_eva_prompt`。

## 数值化转换

如果已经有五维评分 CSV，可以进一步转成数值：

```bash
python score/convert_scores_to_numeric.py \
  --input_csv score/results/results_showo_baseline.csv \
  --output_csv score/results/results_showo_baseline_numeric.csv
```

映射关系：

- `Excellent` -> `5`
- `Good` -> `4`
- `Fair` -> `3`
- `Poor` -> `2`
- `Very Poor` -> `1`

## 结果格式

`score_images.py` 会输出：

- `results/<output_prefix>.json`
- `results/<output_prefix>.csv`

其中每张图片包含 5 个维度评分。

`score_total_images.py` 会输出：

- `results/<output_prefix>.json`
- `results/<output_prefix>.csv`

其中每张图片包含 1 个整体质量分，CSV 最后一行会额外写入平均分。

## 说明

- 脚本只会处理常见图片格式：`png`、`jpg`、`jpeg`、`webp`、`gif`。
- 单张图片超过 20MB 会被跳过并写入日志。
- 如果结果文件已存在，脚本会跳过已经完成评分的图片，适合断点续跑。
