data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# Ingest function: read clips, write decisions, write its own logs. Nothing else.
resource "aws_iam_role" "ingest" {
  name               = "${var.name_prefix}-ingest"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "ingest" {
  statement {
    sid       = "ReadClips"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.clips.arn}/clips/*"]
  }

  statement {
    sid       = "WriteDecisions"
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.decisions.arn]
  }

  statement {
    sid       = "WriteLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.ingest.arn}:*"]
  }
}

resource "aws_iam_role_policy" "ingest" {
  name   = "ingest"
  role   = aws_iam_role.ingest.id
  policy = data.aws_iam_policy_document.ingest.json
}

# API function: read one decision by key, write its own logs.
resource "aws_iam_role" "api" {
  name               = "${var.name_prefix}-api"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "api" {
  statement {
    sid       = "ReadDecisions"
    actions   = ["dynamodb:GetItem"]
    resources = [aws_dynamodb_table.decisions.arn]
  }

  statement {
    sid       = "WriteLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.api.arn}:*"]
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "api"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}
