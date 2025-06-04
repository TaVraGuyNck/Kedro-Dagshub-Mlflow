FROM public.ecr.aws/lambda/python:3.9

# setting working directory
WORKDIR /var/task

# copy requirements
COPY requirements.txt .

# install python dependencies (with cache for speed re-building image if needed)
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy all folders
COPY . .

# setting Lambda handler
CMD ["lambda_function.lambda_handler"]