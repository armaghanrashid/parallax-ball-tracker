"""Static checks on the Terraform. It is never applied, so these guard against drift and typos."""

import re
from pathlib import Path

import pytest

TF_DIR = Path(__file__).resolve().parents[1] / "infra" / "terraform"
SOURCE = "\n".join(p.read_text() for p in sorted(TF_DIR.glob("*.tf")))


def declared(kind: str) -> set[str]:
    return set(re.findall(rf'^{kind} "([^"]+)" "([^"]+)"', SOURCE, flags=re.M))


def test_every_resource_reference_is_declared():
    resources = {f"{t}.{n}" for t, n in declared("resource")} | {
        f"data.{t}.{n}" for t, n in declared("data")
    }
    refs = set(re.findall(r"\b((?:data\.)?aws_[a-z0-9_]+\.[a-z0-9_]+)\b", SOURCE))
    refs = {r for r in refs if not r.startswith("aws_iam_policy_document.")} | {
        r for r in refs if r.startswith("data.")
    }
    missing = {r for r in refs if r not in resources}
    assert not missing, missing


def test_every_variable_reference_is_declared():
    names = set(re.findall(r'^variable "([^"]+)"', SOURCE, flags=re.M))
    used = set(re.findall(r"\bvar\.([a-z_]+)", SOURCE))
    assert used <= names, used - names


@pytest.mark.parametrize(
    "resource",
    [
        "aws_s3_bucket_public_access_block",
        "aws_s3_bucket_server_side_encryption_configuration",
        "aws_s3_bucket_versioning",
        "aws_s3_bucket_notification",
        "aws_dynamodb_table",
        "aws_apigatewayv2_api",
        "aws_apigatewayv2_route",
        "aws_ecr_repository",
    ],
)
def test_required_resources_exist(resource):
    assert any(t == resource for t, _ in declared("resource"))


def test_bucket_is_fully_private_versioned_and_encrypted():
    assert SOURCE.count("= true") >= 4
    for flag in ("block_public_acls", "block_public_policy", "ignore_public_acls"):
        assert re.search(rf"{flag}\s*=\s*true", SOURCE)
    assert 'status = "Enabled"' in SOURCE and "AES256" in SOURCE


def test_dynamodb_is_on_demand():
    assert 'billing_mode = "PAY_PER_REQUEST"' in SOURCE


def test_api_route_and_lambda_image():
    assert 'route_key = "GET /decisions/{id}"' in SOURCE
    assert SOURCE.count('package_type  = "Image"') == 2


def test_iam_is_least_privilege():
    actions = re.findall(r"actions\s*=\s*\[([^\]]*)\]", SOURCE)
    flat = " ".join(actions)
    assert '"*"' not in flat and ":*" not in flat
    assert "s3:PutObject" not in flat and "s3:DeleteObject" not in flat
    resources = re.findall(r"resources\s*=\s*\[([^\]]*)\]", SOURCE)
    assert all('"*"' not in r for r in resources)


def test_provider_and_backend_pinned_without_account_ids():
    assert 'version = "~> 5.0"' in SOURCE and 'source  = "hashicorp/aws"' in SOURCE
    assert 'backend "local"' in SOURCE
    assert not re.search(r"\b\d{12}\b", SOURCE)
