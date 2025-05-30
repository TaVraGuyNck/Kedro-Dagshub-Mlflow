FROM public.ecr.aws/lambda/python:3.9

# Copy the Lambda handler
COPY deployment/handler.py ${LAMBDA_TASK_ROOT}/

# Copy model files
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Copy your custom Kedro package
COPY dist/startupdelay_horizon-0.1-py3-none-any.whl ${LAMBDA_TASK_ROOT}/

# Copy requirements file
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Install your custom package
RUN pip install --no-cache-dir ${LAMBDA_TASK_ROOT}/startupdelay_horizon-0.1-py3-none-any.whl --target "${LAMBDA_TASK_ROOT}"

# Set the Lambda handler function
CMD ["handler.lambda_handler"]