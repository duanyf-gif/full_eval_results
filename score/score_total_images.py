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
MAX_SINGLE_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_IMAGE_DETAIL = "high"
MAX_API_RETRIES = 2
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_SCORE_DIR_NAME = "/root/autodl-tmp/Show-o/Show-O"
DEFAULT_PROMPT_PATH_NAME = "total_eva_prompt"
DEFAULT_IMAGES_DIR_NAME = (
    "/root/autodl-tmp/Show-o/Show-O/eval_results/"
    "itfi_layer23_top50_lam2p0/t2i_images"
)
DEFAULT_OUTPUT_PREFIX = "results_itfi_layer23_top50_lam2p0_total"
DEFAULT_BASE_URL = "https://api.gpt.ge/v1/"

SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-score images with one overall GPT quality score."
    )
    parser.add_argument(
        "--images_dir",
        default=DEFAULT_IMAGES_DIR_NAME,
        help=f"Directory containing images to score. Default: {DEFAULT_IMAGES_DIR_NAME}",
    )
    parser.add_argument(
        "--prompt_path",
        default=DEFAULT_PROMPT_PATH_NAME,
        help=f"Prompt file path. Default: {DEFAULT_PROMPT_PATH_NAME}",
    )
    parser.add_argument(
        "--score_dir",
        default=DEFAULT_SCORE_DIR_NAME,
        help="Directory for results and logs. Default: current Score folder.",
    )
    parser.add_argument(
        "--output_prefix",
        default=DEFAULT_OUTPUT_PREFIX,
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


def resolve_runtime_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def setup_directories(score_dir: Path, output_prefix: str) -> Dict[str, Path]:
    score_dir.mkdir(parents=True, exist_ok=True)
    results_dir = score_dir / "results"
    logs_dir = score_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "score_dir": score_dir,
        "results_json_path": results_dir / f"{output_prefix}.json",
        "results_csv_path": results_dir / f"{output_prefix}.csv",
        "error_log_path": logs_dir / f"{output_prefix}.log",
    }


def configure_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("total_image_scorer")
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
        raise ValueError(f"Prompt file is empty: {prompt_path}")
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
    return {
        "type": "json_schema",
        "name": "overall_image_score",
        "schema": {
            "type": "object",
            "properties": {
                "overall_quality_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["overall_quality_score"],
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


def normalize_total_score(parsed: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None

    score = parsed.get("overall_quality_score")
    if score is None:
        return None

    try:
        numeric_score = int(score)
    except (TypeError, ValueError):
        return None

    if numeric_score < 1 or numeric_score > 10:
        return None

    return {
        "overall_quality_score": numeric_score,
    }


def parse_model_output(raw_text: str) -> Dict[str, Any]:
    for candidate in (raw_text, clean_json_wrappers(raw_text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        normalized = normalize_total_score(parsed)
        if normalized is not None:
            return normalized

    return {
        "overall_quality_score": None,
        "status": "invalid",
        "raw_output": raw_text,
    }


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
            error_text = str(exc)
            if (
                "insufficient_user_quota" in error_text
                or "user quota is not enough" in error_text
            ):
                logger.error("API quota is not enough for %s: %s", image_path, exc)
                return None
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


def save_results(
    results: Dict[str, Dict[str, Any]],
    results_json_path: Path,
    results_csv_path: Path,
) -> None:
    json_results = {
        image_name: {
            "overall_quality_score": result.get("overall_quality_score"),
        }
        for image_name, result in results.items()
    }

    with results_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(json_results, json_file, ensure_ascii=False, indent=2)

    with results_csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_name", "overall_quality_score"],
        )
        writer.writeheader()

        for image_name, result in results.items():
            score = result.get("overall_quality_score")
            writer.writerow(
                {
                    "image_name": image_name,
                    "overall_quality_score": "" if score is None else score,
                }
            )

        valid_scores = [
            result.get("overall_quality_score")
            for result in results.values()
            if isinstance(result.get("overall_quality_score"), int)
        ]
        if valid_scores:
            writer.writerow(
                {
                    "image_name": "average",
                    "overall_quality_score": f"{sum(valid_scores) / len(valid_scores):.4f}",
                }
            )


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
        if results.get(relative_name, {}).get("overall_quality_score") is not None:
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


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be a positive integer.")
    if args.sleep_seconds < 0:
        parser.error("--sleep_seconds must be >= 0.")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    score_dir = resolve_runtime_path(args.score_dir)
    prompt_path = resolve_runtime_path(args.prompt_path)
    images_dir = resolve_runtime_path(args.images_dir)
    score_paths = setup_directories(score_dir, args.output_prefix)
    logger = configure_logger(score_paths["error_log_path"])

    try:
        prompt = load_prompt(prompt_path)
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
