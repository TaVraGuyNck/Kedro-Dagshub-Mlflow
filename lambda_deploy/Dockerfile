FROM public.ecr.aws/lambda/python:3.9

# Install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy everything into image
COPY . .

# Define Lambda handler
CMD ["lambda_function.lambda_handler"]