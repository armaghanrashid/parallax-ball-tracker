variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "parallax"
}

variable "bucket_name" {
  description = "Globally unique name for the clip bucket."
  type        = string
}

variable "image_tag" {
  description = "Tag of the container image pushed to the ECR repository."
  type        = string
  default     = "latest"
}

variable "lambda_memory_mb" {
  description = "Memory for the analysis function (more memory also means more CPU)."
  type        = number
  default     = 2048
}

variable "lambda_timeout_s" {
  description = "Timeout for the analysis function in seconds."
  type        = number
  default     = 120
}

variable "log_retention_days" {
  description = "CloudWatch log retention."
  type        = number
  default     = 14
}
