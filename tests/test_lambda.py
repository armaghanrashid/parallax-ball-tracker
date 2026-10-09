import io
import json
from pathlib import Path

import boto3
import numpy as np
import pytest
from moto import mock_aws

from cloud import lambda_handler
from parallax.clip import Clip, load_clip, save_clip
from parallax.pipeline import analyze
from parallax.sim import simulate
from parallax.sim.render import render

BUCKET = "clips-test-bucket"
TABLE = "parallax-decisions"


@pytest.fixture(scope="module")
def clip_bytes():
    d = simulate("football", 2)
    frames = [render(d, cam) for cam in d.cameras]
    buf = io.BytesIO()
    save_clip(buf, Clip(frames=np.stack(frames), cameras=d.cameras, sport="football"))
    return d, frames, buf.getvalue()


@pytest.fixture
def aws(monkeypatch):
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "DECISIONS_TABLE": TABLE,
    }.items():
        monkeypatch.setenv(k, v)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket=BUCKET)
        boto3.client("dynamodb").create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield


def s3_event(key):
    return {"Records": [{"s3": {"bucket": {"name": BUCKET}, "object": {"key": key}}}]}


def test_clip_round_trip(clip_bytes):
    d, _, raw = clip_bytes
    clip = load_clip(io.BytesIO(raw))
    assert clip.sport == "football" and clip.fps == 240.0
    assert clip.frames.shape[0] == 2
    assert pytest.approx(d.cameras[0].P) == clip.cameras[0].P


def test_s3_upload_to_dynamodb_end_to_end(aws, clip_bytes):
    d, frames, raw = clip_bytes
    boto3.client("s3").put_object(Bucket=BUCKET, Key="clips/match-17.npz", Body=raw)

    out = lambda_handler.handler(s3_event("clips/match-17.npz"), None)

    expected = analyze(frames, d.cameras, "football")
    assert out == {"processed": [{"id": "match-17", "decision": expected.decision.label}]}
    item = boto3.resource("dynamodb").Table(TABLE).get_item(Key={"id": "match-17"})["Item"]
    assert item["decision"] == expected.decision.label == d.truth.label
    assert item["sport"] == "football"
    assert item["source"] == f"s3://{BUCKET}/clips/match-17.npz"
    assert float(item["margin_mm"]) == pytest.approx(expected.decision.margin_mm, abs=1e-3)
    assert int(item["n_points"]) > 20


def test_keys_with_url_encoding_are_decoded(aws, clip_bytes):
    boto3.client("s3").put_object(Bucket=BUCKET, Key="clips/a b.npz", Body=clip_bytes[2])
    out = lambda_handler.handler(s3_event("clips/a+b.npz"), None)
    assert out["processed"][0]["id"] == "a b"


def test_api_returns_stored_decision_and_404_for_unknown(aws, clip_bytes):
    boto3.client("s3").put_object(Bucket=BUCKET, Key="clips/c1.npz", Body=clip_bytes[2])
    lambda_handler.handler(s3_event("clips/c1.npz"), None)

    ok = lambda_handler.api_handler({"pathParameters": {"id": "c1"}}, None)
    body = json.loads(ok["body"])
    assert (
        ok["statusCode"] == 200 and body["id"] == "c1" and body["decision"] in {"GOAL", "NO_GOAL"}
    )

    missing = lambda_handler.api_handler({"pathParameters": {"id": "nope"}}, None)
    assert missing["statusCode"] == 404
    assert lambda_handler.api_handler({}, None)["statusCode"] == 400


def test_pipeline_modules_do_not_import_aws():
    import parallax.pipeline
    import parallax.vision.fit  # noqa: F401

    src = Path(parallax.pipeline.__file__).read_text()
    assert "boto3" not in src
