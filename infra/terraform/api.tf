resource "aws_apigatewayv2_api" "decisions" {
  name          = "${var.name_prefix}-decisions"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "get_decision" {
  api_id                 = aws_apigatewayv2_api.decisions.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "get_decision" {
  api_id    = aws_apigatewayv2_api.decisions.id
  route_key = "GET /decisions/{id}"
  target    = "integrations/${aws_apigatewayv2_integration.get_decision.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.decisions.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "allow_apigw" {
  statement_id  = "AllowApiGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.decisions.execution_arn}/*/*"
}
