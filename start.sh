gunicorn --bind 0.0.0.0:8000 --workers=10 -t 200 wsgi:app 
