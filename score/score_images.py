from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from openai import OpenAI


SUPPORTED_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
SCORE_KEYS = [
    "brushstroke",
    "color",
    "composition",
    "light_and_shadow",
    "line_quality",
]
MAX_SINGLE_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_IMAGE_DETAIL = "high"
MAX_API_RETRIES = 2
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_SCORE_DIR_NAME = "."
DEFAULT_IMAGES_DIR_NAME = "../eval_results/emoart_janus_itfi_layer23_top100_lam1p0_per_category/t2i_images"
DEFAULT_BASE_URL = "https://api.gpt.ge/v1/"
#EmoArt-5k_local/emoart_val/Images/Art Brut
#eval_results/baseline/t2i_images
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = """You are an expert art critic specializing in visual analysis of paintings.
Evaluate the artistic quality of the given image based on the following five visual attributes:
1. Brushstroke – the visibility, richness, and expressiveness of painting strokes.
2. Color – the harmony, richness, and artistic use of colors.
3. Composition – the arrangement and balance of visual elements.
4. Light and Shadow – the effectiveness of lighting and contrast.
5. Line Quality – the clarity, strength, and dynamism of lines.
For each attribute, assign one rating from the following levels:
Excellent
Good
Fair
Poor
Very Poor
Rating meaning:
Excellent – outstanding artistic quality  
Good – strong and convincing  
Fair – acceptable but unremarkable  
Poor – weak execution  
Very Poor – extremely weak or absent
Important rules:
- Evaluate each attribute independently.
- Focus on artistic qualities rather than object realism.
- Return only the ratings.
Output format:
{
 "brushstroke": "rating",
 "color": "rating",
 "composition": "rating",
 "light_and_shadow": "rating",
 "line_quality": "rating"
}

"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-score images with the OpenAI Responses API."
    )
    parser.add_argument(
        "--images_dir",
        default=DEFAULT_IMAGES_DIR_NAME,
        help=f"Directory containing images to score. Default: {DEFAULT_IMAGES_DIR_NAME}",
    )
    parser.add_argument(
        "--score_dir",
        default=DEFAULT_SCORE_DIR_NAME,
        help="Directory for prompt, results, and logs. Default: the current Score folder.",
    )
    parser.add_argument(
        "--output_prefix",
        default="results_janus_itfi_layer23_top100_lam1p0",
        help="Base filename for JSON/CSV results and log names, without extension.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process. Default: all images.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Responses API model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help=f"Optional OpenAI-compatible base URL. If omitted, use OPENAI_BASE_URL or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Seconds to sleep after each API request. Default: {DEFAULT_SLEEP_SECONDS}",
    )
    parser.add_argument(
        "--image_detail",
        choices=["low", "high", "original", "auto"],
        default=DEFAULT_IMAGE_DETAIL,
        help=f"Detail level sent with input_image. Default: {DEFAULT_IMAGE_DETAIL}",
    )
    return parser


def setup_directories(score_dir: Path, output_prefix: str) -> Dict[str, Path]:
    score_dir.mkdir(parents=True, exist_ok=True)
    results_dir = score_dir / "results"
    logs_dir = score_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = score_dir / "prompt.txt"
    if not prompt_path.exists():
        prompt_path.write_text(DEFAULT_PROMPT, encoding="utf-8")

    return {
        "score_dir": score_dir,
        "prompt_path": prompt_path,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "results_json_path": results_dir / f"{output_prefix}.json",
        "results_csv_path": results_dir / f"{output_prefix}.csv",
        "error_log_path": logs_dir / f"{output_prefix}.log",
    }


def configure_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("image_scorer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_prompt(prompt_path: Path) -> str:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(
            f"Prompt file is empty: {prompt_path}. Please fill it before running."
        )
    return prompt


def iter_image_files(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_TYPES:
            yield path


def image_to_data_url(image_path: Path) -> str:
    mime_type = SUPPORTED_IMAGE_TYPES[image_path.suffix.lower()]
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def response_schema() -> Dict[str, Any]:
    properties = {key: {"type": "string"} for key in SCORE_KEYS}
    return {
        "type": "json_schema",
        "name": "image_scores",
        "schema": {
            "type": "object",
            "properties": properties,
            "required": SCORE_KEYS,
            "additionalProperties": False,
        },
        "strict": True,
    }


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output = getattr(response, "output", None)
    if output:
        chunks: list[str] = []
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(text)
        if chunks:
            return "\n".join(chunks)

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    return str(response)


def clean_json_wrappers(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        return cleaned[first_brace : last_brace + 1]

    return cleaned


def normalize_scores(parsed: Any) -> Optional[Dict[str, str]]:
    if isinstance(parsed, dict) and "scores" in parsed and isinstance(parsed["scores"], dict):
        parsed = parsed["scores"]

    if not isinstance(parsed, dict):
        return None

    normalized: Dict[str, str] = {}
    for key in SCORE_KEYS:
        value = parsed.get(key)
        if value is None:
            return None
        normalized[key] = str(value).strip()
    return normalized


def parse_model_output(raw_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
        normalized = normalize_scores(parsed)
        if normalized is not None:
            return normalized
    except json.JSONDecodeError:
        pass

    cleaned = clean_json_wrappers(raw_text)
    try:
        parsed = json.loads(cleaned)
        normalized = normalize_scores(parsed)
        if normalized is not None:
            return normalized
    except json.JSONDecodeError:
        pass

    invalid_result: Dict[str, Any] = {key: None for key in SCORE_KEYS}
    invalid_result["status"] = "invalid"
    invalid_result["raw_output"] = raw_text
    return invalid_result


def evaluate_image(
    image_path: Path,
    prompt: str,
    client: OpenAI,
    logger: logging.Logger,
    model: str = DEFAULT_MODEL,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    image_detail: str = DEFAULT_IMAGE_DETAIL,
) -> Optional[Dict[str, Any]]:
    try:
        file_size = image_path.stat().st_size
    except OSError as exc:
        logger.error("Failed to read file size for %s: %s", image_path, exc)
        return None

    if file_size > MAX_SINGLE_IMAGE_BYTES:
        logger.error(
            "Skipped %s because file size exceeds 20MB (%.2f MB).",
            image_path,
            file_size / (1024 * 1024),
        )
        return None

    try:
        image_data_url = image_to_data_url(image_path)
    except OSError as exc:
        logger.error("Failed to read image %s: %s", image_path, exc)
        return None

    last_error: Optional[Exception] = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": image_data_url,
                                "detail": image_detail,
                            },
                        ],
                    }
                ],
                text={
                    "verbosity": "low",
                    "format": response_schema(),
                },
            )
            raw_output = extract_response_text(response)
            return parse_model_output(raw_output)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_API_RETRIES:
                logger.warning(
                    "API call failed for %s (attempt %s/%s): %s",
                    image_path,
                    attempt + 1,
                    MAX_API_RETRIES + 1,
                    exc,
                )
            else:
                logger.error(
                    "API call failed for %s after %s attempts: %s",
                    image_path,
                    MAX_API_RETRIES + 1,
                    exc,
                )
        finally:
            time.sleep(sleep_seconds)

    if last_error:
        logger.debug("Final failure for %s: %r", image_path, last_error)
    return None


def load_existing_results(
    results_json_path: Path,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    if not results_json_path.exists():
        return {}

    try:
        with results_json_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
        if isinstance(data, dict):
            return data
        logger.warning("Existing results file is not a JSON object: %s", results_json_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load existing results from %s: %s", results_json_path, exc)

    return {}


def process_images(
    images_dir: Path,
    prompt: str,
    logger: logging.Logger,
    max_images: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    image_detail: str = DEFAULT_IMAGE_DETAIL,
    existing_results: Optional[Dict[str, Dict[str, Any]]] = None,
    results_json_path: Optional[Path] = None,
    results_csv_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    image_paths = list(iter_image_files(images_dir))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    effective_base_url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    client_kwargs: Dict[str, Any] = {"timeout": timeout}
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url
    client = OpenAI(**client_kwargs)
    results: Dict[str, Dict[str, Any]] = dict(existing_results or {})

    pending_image_paths = []
    for image_path in image_paths:
        relative_name = image_path.relative_to(images_dir).as_posix()
        if relative_name in results:
            continue
        pending_image_paths.append(image_path)

    print(f"Using model: {model}")
    print(f"Using base URL: {effective_base_url or 'https://api.openai.com/v1'}")
    print(f"Resolved images directory: {images_dir}")
    print(f"Loaded existing results: {len(results)}")
    print(f"Images pending scoring: {len(pending_image_paths)}")

    for index, image_path in enumerate(pending_image_paths, start=1):
        relative_name = image_path.relative_to(images_dir).as_posix()
        print(f"Processing image {index}/{len(pending_image_paths)}... {relative_name}")

        try:
            result = evaluate_image(
                image_path=image_path,
                prompt=prompt,
                client=client,
                logger=logger,
                model=model,
                sleep_seconds=sleep_seconds,
                image_detail=image_detail,
            )
            if result is not None:
                results[relative_name] = result
                if results_json_path is not None and results_csv_path is not None:
                    save_results(
                        results=results,
                        results_json_path=results_json_path,
                        results_csv_path=results_csv_path,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while processing %s: %s", image_path, exc)

    return results


def save_results(
    results: Dict[str, Dict[str, Any]],
    results_json_path: Path,
    results_csv_path: Path,
) -> None:
    with results_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    with results_csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_name", *SCORE_KEYS])
        writer.writeheader()

        for image_name, result in results.items():
            row = {"image_name": image_name}
            for key in SCORE_KEYS:
                value = result.get(key)
                row[key] = "" if value is None else value
            writer.writerow(row)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be a positive integer.")
    if args.sleep_seconds < 0:
        parser.error("--sleep_seconds must be >= 0.")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0.")


def resolve_runtime_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    score_dir = resolve_runtime_path(args.score_dir)
    images_dir = resolve_runtime_path(args.images_dir)
    score_paths = setup_directories(score_dir, args.output_prefix)
    logger = configure_logger(score_paths["error_log_path"])
    try:
        prompt = load_prompt(score_paths["prompt_path"])
        existing_results = load_existing_results(score_paths["results_json_path"], logger)
        existing_count = len(existing_results)

        results = process_images(
            images_dir=images_dir,
            prompt=prompt,
            logger=logger,
            max_images=args.max_images,
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            image_detail=args.image_detail,
            existing_results=existing_results,
            results_json_path=score_paths["results_json_path"],
            results_csv_path=score_paths["results_csv_path"],
        )

        save_results(
            results=results,
            results_json_path=score_paths["results_json_path"],
            results_csv_path=score_paths["results_csv_path"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Script execution failed: %s", exc)
        print(f"Script failed. Check log: {score_paths['error_log_path']}", file=sys.stderr)
        sys.exit(1)

    print(f"Newly processed image result(s): {len(results) - existing_count}")
    print(f"Total saved image result(s): {len(results)}")
    print(f"Saved JSON results to: {score_paths['results_json_path']}")
    print(f"Saved CSV results to: {score_paths['results_csv_path']}")
    print(f"Error log path: {score_paths['error_log_path']}")


if __name__ == "__main__":
    main()
