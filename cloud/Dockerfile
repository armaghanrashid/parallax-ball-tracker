# Lambda container image. Build from the repository root:
#   docker build -f cloud/Dockerfile -t parallax-ball-tracker .
FROM public.ecr.aws/lambda/python:3.12

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY parallax ${LAMBDA_TASK_ROOT}/parallax
COPY cloud ${LAMBDA_TASK_ROOT}/cloud

# The ingest function uses this handler; the API function overrides it with
# cloud.lambda_handler.api_handler through image_config in Terraform.
CMD ["cloud.lambda_handler.handler"]
