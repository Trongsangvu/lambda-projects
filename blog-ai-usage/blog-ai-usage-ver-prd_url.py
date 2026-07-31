import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
import time
import os

# https://api.v-m-ai.com

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ============================================================
# Configuration
# ============================================================

_OPENAI_USAGE_URL = os.environ.get("OPENAI_USAGE_URL", "").strip().rstrip("/")

_DEFAULT_LOOK_BACK_DAYS = 14
_TIME_OUT_SECONDS = 60

_MAX_RETRIES = 4

_RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# Global request counter
# ============================================================

_openai_request_count = 0


# ============================================================
# Helpers
# ============================================================


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(dt: datetime) -> int:
    return int(dt.timestamp())


def _get_retry_delay(
    error: urllib.error.HTTPError,
    attempt: int,
) -> float:
    """
    Get retry delay from Retry-After header if available.
    Otherwise use exponential backoff with a maximum of 60 seconds.
    For 429 Rate Limits, use higher initial backoff.
    """

    retry_after = error.headers.get("Retry-After")

    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    if error.code == 429:
        # 15s → 30s → 60s → 60s
        return min(60.0, 15.0 * (2**attempt))

    # 10s → 20s → 40s → 60s
    return min(60.0, 10.0 * (2**attempt))


def _urlopen_with_retry(
    req: urllib.request.Request,
    timeout: int = _TIME_OUT_SECONDS,
):
    """
    Execute HTTP request with retry support for:
        429
        500
        502
        503
        504
    """

    global _openai_request_count

    for attempt in range(_MAX_RETRIES + 1):
        try:
            _openai_request_count += 1

            logger.info(
                f"OpenAI request #{_openai_request_count} "
                f"attempt={attempt + 1}/{_MAX_RETRIES + 1} "
                f"url={req.full_url}"
            )

            return urllib.request.urlopen(
                req,
                timeout=timeout,
            )

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)

            logger.warning(
                f"OpenAI HTTP error "
                f"status={e.code} "
                f"attempt={attempt + 1}/{_MAX_RETRIES + 1} "
                f"url={req.full_url} "
                f"body={error_body}"
            )

            # Non-retryable error
            if e.code not in _RETRYABLE_STATUS_CODES:
                raise

            # No more retry
            if attempt >= _MAX_RETRIES:
                logger.error(
                    f"OpenAI request failed after "
                    f"{_MAX_RETRIES + 1} attempts. "
                    f"status={e.code}"
                )
                raise

            delay = _get_retry_delay(e, attempt)

            logger.warning(
                f"Retrying OpenAI request in {delay:.2f}s " f"because of HTTP {e.code}"
            )

            time.sleep(delay)

        except (
            urllib.error.URLError,
            TimeoutError,
        ) as e:

            logger.warning(
                f"OpenAI network error "
                f"attempt={attempt + 1}/{_MAX_RETRIES + 1}: "
                f"{str(e)}"
            )

            if attempt >= _MAX_RETRIES:
                raise

            delay = min(60, 2**attempt)

            logger.warning(f"Retrying OpenAI request in {delay:.2f}s")

            time.sleep(delay)

    raise RuntimeError("OpenAI request failed")


# ============================================================
# OpenAI Costs API
# ============================================================


def _fetch_openai_costs(admin_key: str) -> list:
    """
    Fetch OpenAI costs for the configured lookback period.
    """

    now_dt = _utc_now()

    start_dt = now_dt - timedelta(days=_DEFAULT_LOOK_BACK_DAYS)

    start_time = _timestamp(start_dt)
    end_time = _timestamp(now_dt)

    url = (
        f"{_OPENAI_USAGE_URL}/costs"
        f"?start_time={start_time}"
        f"&end_time={end_time}"
        f"&bucket_width=1d"
        f"&limit=100"
    )

    headers = {
        "Authorization": f"Bearer {admin_key}",
        "Content-Type": "application/json",
    }

    logger.info(
        "Fetching OpenAI costs: "
        f"start={start_dt.isoformat()} "
        f"end={now_dt.isoformat()}"
    )

    req = urllib.request.Request(
        url,
        headers=headers,
        method="GET",
    )

    try:
        with _urlopen_with_retry(req) as res:
            response_data = json.loads(res.read().decode("utf-8"))

        data = response_data.get("data", [])

        logger.info(f"Fetched {len(data)} OpenAI cost buckets")

        return data

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)

        logger.error(f"OpenAI costs API error " f"{e.code}: {error_body}")

        raise

    except Exception as e:
        logger.error(
            f"Failed to fetch OpenAI costs: {str(e)}",
            exc_info=True,
        )

        raise


# ============================================================
# OpenAI Completions Usage API
# ============================================================


def _fetch_openai_completions(admin_key: str) -> list:
    """
    Fetch OpenAI completions usage for the configured lookback period.

    Supports pagination through next_page.
    """

    now_dt = _utc_now()

    # Start at midnight UTC
    end_dt = now_dt.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    start_dt = end_dt - timedelta(days=_DEFAULT_LOOK_BACK_DAYS)

    start_time = _timestamp(start_dt)
    end_time = _timestamp(end_dt)

    base_url = (
        f"{_OPENAI_USAGE_URL}/usage/completions"
        f"?start_time={start_time}"
        f"&end_time={end_time}"
    )

    headers = {
        "Authorization": f"Bearer {admin_key}",
        "Content-Type": "application/json",
    }

    all_buckets = []

    page_token = None
    page_count = 0

    while True:
        page_count += 1

        current_url = base_url

        if page_token:
            current_url = f"{base_url}&page={page_token}"

        logger.info(f"Fetching OpenAI completions " f"page={page_count}")

        req = urllib.request.Request(
            current_url,
            headers=headers,
            method="GET",
        )

        try:
            with _urlopen_with_retry(req) as res:
                response_data = json.loads(res.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)

            logger.error(f"OpenAI completions API error " f"{e.code}: {error_body}")

            raise

        except Exception as e:
            logger.error(
                f"Failed to fetch OpenAI completions: " f"{str(e)}",
                exc_info=True,
            )

            raise

        page_data = response_data.get("data", [])

        logger.info(
            f"Completion page={page_count} " f"returned {len(page_data)} buckets"
        )

        all_buckets.extend(page_data)

        has_more = response_data.get(
            "has_more",
            False,
        )

        next_page = response_data.get("next_page")

        if has_more and next_page:
            page_token = next_page

            logger.info(f"More completion pages available. " f"next_page={page_token}")
            time.sleep(1.5)
        else:
            break

    logger.info(
        f"Finished fetching OpenAI completions. "
        f"pages={page_count}, "
        f"total_buckets={len(all_buckets)}"
    )

    return all_buckets


# ============================================================
# Snapshot Builder
# ============================================================


def _build_snapshot(
    cost_data: list,
    completions_data: list,
) -> dict:

    usage_daily = []
    currency = "usd"

    # --------------------------------------------------------
    # Cost daily
    # --------------------------------------------------------

    for bucket in cost_data:

        amount = round(
            sum(
                float(
                    result.get(
                        "amount",
                        {},
                    ).get(
                        "value",
                        0,
                    )
                )
                for result in bucket.get(
                    "results",
                    [],
                )
            ),
            2,
        )

        if bucket.get("results"):
            currency = bucket["results"][0].get("amount", {}).get("currency", "usd")

        bucket_start_time = bucket.get("start_time")

        if bucket_start_time is None:
            continue

        bucket_date = datetime.fromtimestamp(
            bucket_start_time,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d")

        usage_daily.append(
            {
                "date": bucket_date,
                "amount": amount,
            }
        )

    # --------------------------------------------------------
    # Date ranges
    # --------------------------------------------------------

    today = _utc_now().date()

    lookback_start_date = today - timedelta(days=_DEFAULT_LOOK_BACK_DAYS)

    current_month_start_date = today.replace(day=1)

    # --------------------------------------------------------
    # 14-day lookback total
    # --------------------------------------------------------

    lookback_total = round(
        sum(
            entry["amount"]
            for entry in usage_daily
            if datetime.strptime(
                entry["date"],
                "%Y-%m-%d",
            ).date()
            >= lookback_start_date
        ),
        2,
    )

    # --------------------------------------------------------
    # Current month cost
    # --------------------------------------------------------

    current_month_cost = round(
        sum(
            entry["amount"]
            for entry in usage_daily
            if datetime.strptime(
                entry["date"],
                "%Y-%m-%d",
            ).date()
            >= current_month_start_date
        ),
        2,
    )

    # --------------------------------------------------------
    # Completions daily
    # --------------------------------------------------------

    completions_daily = []

    for bucket in completions_data:

        bucket_start_time = bucket.get("start_time")

        if bucket_start_time is None:
            continue

        results = bucket.get(
            "results",
            [],
        )

        completions_daily.append(
            {
                "date": datetime.fromtimestamp(
                    bucket_start_time,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d"),
                "input_tokens": sum(
                    int(
                        result.get(
                            "input_tokens",
                            0,
                        )
                    )
                    for result in results
                ),
                "input_cached_tokens": sum(
                    int(
                        result.get(
                            "input_cached_tokens",
                            0,
                        )
                    )
                    for result in results
                ),
                "input_audio_tokens": sum(
                    int(
                        result.get(
                            "input_audio_tokens",
                            0,
                        )
                    )
                    for result in results
                ),
                "output_tokens": sum(
                    int(
                        result.get(
                            "output_tokens",
                            0,
                        )
                    )
                    for result in results
                ),
                "output_audio_tokens": sum(
                    int(
                        result.get(
                            "output_audio_tokens",
                            0,
                        )
                    )
                    for result in results
                ),
                "num_model_requests": sum(
                    int(
                        result.get(
                            "num_model_requests",
                            0,
                        )
                    )
                    for result in results
                ),
            }
        )

    # --------------------------------------------------------
    # Build snapshot
    # --------------------------------------------------------

    snapshot = {
        "usage": {
            "total": lookback_total,
            "currency": currency,
            "daily": usage_daily,
        },
        "current_month_cost": current_month_cost,
        "completions": {
            "total_input_tokens": sum(d["input_tokens"] for d in completions_daily),
            "total_input_cached_tokens": sum(
                d["input_cached_tokens"] for d in completions_daily
            ),
            "total_input_audio_tokens": sum(
                d["input_audio_tokens"] for d in completions_daily
            ),
            "total_output_tokens": sum(d["output_tokens"] for d in completions_daily),
            "total_output_audio_tokens": sum(
                d["output_audio_tokens"] for d in completions_daily
            ),
            "total_num_model_requests": sum(
                d["num_model_requests"] for d in completions_daily
            ),
            "daily": completions_daily,
        },
    }

    return snapshot


# ============================================================
# Post Snapshot
# ============================================================


def _post_snapshot(
    url: str,
    api_key: str,
    snapshot: dict,
):
    target_url = f"{url}/api/v1/usage-api/openai/snapshot"

    logger.info(f"Posting snapshot to {target_url}...")

    snapshot_json = json.dumps(snapshot).encode("utf-8")

    req = urllib.request.Request(
        target_url,
        data=snapshot_json,
        method="POST",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=_TIME_OUT_SECONDS,
    ) as res:

        status_code = res.getcode()

        body = res.read().decode("utf-8")

        return status_code, body


# ============================================================
# Lambda Handler
# ============================================================


def lambda_handler(event, context):

    global _openai_request_count

    # Reset request counter for this invocation
    _openai_request_count = 0

    request_id = getattr(
        context,
        "aws_request_id",
        "unknown",
    )

    logger.info("==================================================")

    logger.info(f"OpenAI Usage Snapshot Lambda started. " f"request_id={request_id}")

    logger.info(f"Lookback days: {_DEFAULT_LOOK_BACK_DAYS}")

    # --------------------------------------------------------
    # Environment variables
    # --------------------------------------------------------

    api_url = (
        os.environ.get(
            "API_URL",
            "",
        )
        .strip()
        .rstrip("/")
    )

    api_key = (
        os.environ.get(
            "API_KEY",
            "",
        )
        .strip()
        .strip("\"'")
    )

    openai_admin_key = (
        os.environ.get(
            "OPENAI_ADMIN_KEY",
            "",
        )
        .strip()
        .strip("\"'")
    )

    # --------------------------------------------------------
    # Validate environment
    # --------------------------------------------------------

    if not api_url or not api_key or not openai_admin_key:

        logger.error(
            "Missing required environment variables. "
            "Required: "
            "API_URL, "
            "API_KEY, "
            "OPENAI_ADMIN_KEY"
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": ("Missing required environment " "variables")}
            ),
        }

    try:

        # ====================================================
        # 1. Fetch OpenAI Costs
        # ====================================================

        logger.info("Step 1/4: Fetching OpenAI cost usage...")

        cost_data = _fetch_openai_costs(openai_admin_key)

        logger.info(f"Fetched {len(cost_data)} cost buckets")

        # Brief pause between requests to prevent hitting OpenAI rate limits (30 req/min)
        time.sleep(1.0)

        # ====================================================
        # 2. Fetch OpenAI Completions
        # ====================================================

        logger.info("Step 2/4: Fetching OpenAI completions usage...")

        completions_data = _fetch_openai_completions(openai_admin_key)

        logger.info(f"Fetched " f"{len(completions_data)} " f"completion buckets")

        # ====================================================
        # 3. Build snapshot
        # ====================================================

        logger.info("Step 3/4: Building snapshot...")

        snapshot = _build_snapshot(
            cost_data,
            completions_data,
        )

        logger.info("Snapshot built successfully.")

        logger.info(
            f"Current month cost: "
            f"{snapshot['current_month_cost']} "
            f"{snapshot['usage']['currency']}"
        )

        logger.info(
            f"14-day lookback cost: "
            f"{snapshot['usage']['total']} "
            f"{snapshot['usage']['currency']}"
        )

        logger.info(
            f"Total model requests: "
            f"{snapshot['completions']['total_num_model_requests']}"
        )

        # ====================================================
        # 4. Post snapshot to backend
        # ====================================================

        logger.info("Step 4/4: Posting snapshot...")

        status_code, response_body = _post_snapshot(
            api_url,
            api_key,
            snapshot,
        )

        logger.info(f"Server response " f"({status_code}): " f"{response_body}")

        logger.info(
            f"Lambda completed successfully. "
            f"OpenAI requests made: "
            f"{_openai_request_count}"
        )

        logger.info("==================================================")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": (f"Sync completed via {api_url}"),
                    "openai_requests": (_openai_request_count),
                    "data": snapshot,
                }
            ),
        }

    except urllib.error.HTTPError as e:

        error_body = e.read().decode("utf-8") if e.fp else str(e)

        logger.error(f"HTTP error " f"{e.code}: {error_body}")

        return {
            "statusCode": e.code,
            "body": json.dumps(
                {
                    "error": error_body,
                    "openai_requests": (_openai_request_count),
                }
            ),
        }

    except Exception as e:

        logger.error(
            f"Sync failed: {str(e)}",
            exc_info=True,
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": str(e),
                    "openai_requests": (_openai_request_count),
                }
            ),
        }
