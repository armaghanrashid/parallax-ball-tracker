output "clip_bucket" {
  description = "Upload clip .npz files under clips/ in this bucket."
  value       = aws_s3_bucket.clips.bucket
}

output "ecr_repository_url" {
  description = "Push the Lambda container image here."
  value       = aws_ecr_repository.pipeline.repository_url
}

output "decisions_table" {
  value = aws_dynamodb_table.decisions.name
}

output "api_url" {
  description = "Base URL of the HTTP API; GET <api_url>/decisions/{id}."
  value       = aws_apigatewayv2_api.decisions.api_endpoint
}
