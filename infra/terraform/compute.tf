locals {
  image_uri = "${aws_ecr_repository.pipeline.repository_url}:${var.image_tag}"
}

resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/aws/lambda/${var.name_prefix}-ingest"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${var.name_prefix}-api"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "ingest" {
  function_name = "${var.name_prefix}-ingest"
  role          = aws_iam_role.ingest.arn
  package_type  = "Image"
  image_uri     = local.image_uri
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_s

  image_config {
    command = ["cloud.lambda_handler.handler"]
  }

  environment {
    variables = {
      DECISIONS_TABLE = aws_dynamodb_table.decisions.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.ingest]
}

resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.clips.arn
}

resource "aws_lambda_function" "api" {
  function_name = "${var.name_prefix}-api"
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = local.image_uri
  memory_size   = 256
  timeout       = 10

  image_config {
    command = ["cloud.lambda_handler.api_handler"]
  }

  environment {
    variables = {
      DECISIONS_TABLE = aws_dynamodb_table.decisions.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.api]
}
