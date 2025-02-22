import os

from rhizotron.settings.defaults import *

SECRET_KEY = os.environ.get(
    'SECRET_KEY', '769!fm4a_2+8g^o^o4ijwd4-z963=zxp=dj-^#&%7#o1uv-qa6')

DEBUG = True
CORS_ORIGIN_ALLOW_ALL = DEBUG

ALLOWED_HOSTS = ['*']

CSRF_TRUSTED_ORIGINS = ['http://localhost:8000']

DATABASES = {
    'default': {
        'ENGINE': 'django_prometheus.db.backends.mysql',
        'NAME': os.environ.get('DATABASE_NAME'),
        'USER': os.environ.get('DATABASE_USERNAME'),
        'PASSWORD': os.environ.get('DATABASE_PASSWORD'),
        'HOST': os.environ.get('DATABASE_HOST'),
        'PORT': '3306'
    }
}

Q_CLUSTER['redis'] = {
    'host': os.environ.get("QUEUE_HOST"),
    'port': 6379,
    'db': 0,
    'password': None,
    'socket_timeout': None,
    'charset': 'utf-8',
    'errors': 'strict',
    'unix_socket_path': None
}
