from .local import *

DATABASES = {
    'default': {
        'ENGINE': 'django_prometheus.db.backends.mysql',
        'NAME': 'dev',
        'USER': 'root',
        'PASSWORD': 'passw0rd',
        'HOST': 'mysqldb',
        'PORT': '3306'
    }
}

# Ensure CORS is properly configured for Docker environment
CORS_ORIGIN_ALLOW_ALL = True
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ['*']
CORS_ALLOW_HEADERS = ['*']
CORS_EXPOSE_HEADERS = ['*']
CORS_PREFLIGHT_MAX_AGE = 86400

# Disable CSRF protection
CSRF_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = False
CSRF_USE_SESSIONS = False
CSRF_COOKIE_SAMESITE = None

# Session settings
SESSION_COOKIE_SECURE = False
SESSION_COOKIE_HTTPONLY = False
SESSION_COOKIE_SAMESITE = None

Q_CLUSTER['redis'] = {
    'host': 'redis',
    'port': 6379,
    'db': 0,
    'password': None,
    'socket_timeout': None,
    'charset': 'utf-8',
    'errors': 'strict',
    'unix_socket_path': None
}
