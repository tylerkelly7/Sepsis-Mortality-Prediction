install:
\tpip install -r requirements.txt

test:
\tpytest tests/

lint:
\tflake8 src tests

format:
\tblack src tests

run:
\tpython app/app.py
