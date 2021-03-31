
# Access log - records incoming HTTP requests
accesslog = "./app/log/gunicorn.access.log"
# Error log - records Gunicorn server goings-on
errorlog = "./app/log/gunicorn.error.log"
# How verbose the Gunicorn error logs should be 
loglevel = "info"