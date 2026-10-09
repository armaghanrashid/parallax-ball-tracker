"""AWS Lambda entry points.

`handler`      S3 ObjectCreated -> run the pipeline on the uploaded clip -> write the decision to
               DynamoDB (table name from the DECISIONS_TABLE environment variable).
`api_handler`  API Gateway HTTP API GET /decisions/{id} -> read one decision back.

The vision pipeline itself (`parallax.*`) has no AWS dependency.
"""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import unquote_plus

import boto3

from parallax.clip import load_clip
from parallax.pipeline import analyze

DEFAULT_TABLE = "parallax-decisions"


def _table():
    return boto3.resource("dynamodb").Table(os.environ.get("DECISIONS_TABLE", DEFAULT_TABLE))


def _dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(round(float(value), 4)))


def process_object(bucket: str, key: str) -> dict:
    """Analyse one clip stored in S3 and persist the decision. Returns the stored item."""
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    clip = load_clip(io.BytesIO(body))
    result = analyze(clip.frames, clip.cameras, clip.sport, clip.fps)
    decision = result.decision
    item = {
        "id": os.path.splitext(os.path.basename(key))[0],
        "sport": clip.sport,
        "decision": decision.label,
        "margin_mm": _dec(decision.margin_mm),
        "impact_a": _dec(decision.impact[0]) if decision.impact else None,
        "impact_b": _dec(decision.impact[1]) if decision.impact else None,
        "n_points": int(len(result.points)),
        "n_frames": int(result.n_frames),
        "latency_ms": _dec(result.latency_ms),
        "source": f"s3://{bucket}/{key}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    item = {k: v for k, v in item.items() if v is not None}
    _table().put_item(Item=item)
    return item


def handler(event, context=None) -> dict:
    processed = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        item = process_object(bucket, key)
        processed.append({"id": item["id"], "decision": item["decision"]})
    return {"processed": processed}


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(type(value))


def _response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload, default=_json_default),
    }


def api_handler(event, context=None) -> dict:
    decision_id = (event.get("pathParameters") or {}).get("id")
    if not decision_id:
        return _response(400, {"error": "missing id"})
    item = _table().get_item(Key={"id": decision_id}).get("Item")
    if item is None:
        return _response(404, {"error": "not found", "id": decision_id})
    return _response(200, item)
