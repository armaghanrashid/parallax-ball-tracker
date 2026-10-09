resource "aws_s3_bucket" "clips" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "clips" {
  bucket                  = aws_s3_bucket.clips.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "clips" {
  bucket = aws_s3_bucket.clips.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_versioning" "clips" {
  bucket = aws_s3_bucket.clips.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_notification" "clip_uploaded" {
  bucket = aws_s3_bucket.clips.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingest.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "clips/"
    filter_suffix       = ".npz"
  }

  depends_on = [aws_lambda_permission.allow_s3]
}

resource "aws_dynamodb_table" "decisions" {
  name         = "${var.name_prefix}-decisions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }

  server_side_encryption {
    enabled = true
  }

  point_in_time_recovery {
    enabled = true
  }
}

resource "aws_ecr_repository" "pipeline" {
  name                 = "${var.name_prefix}-ball-tracker"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
