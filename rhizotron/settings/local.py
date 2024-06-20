from .defaults import *


DEBUG = True

ALLOWED_HOSTS = ['*']

DATABASES = {
    'default': {
        'ENGINE': 'django_prometheus.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs/run.log',
            'formatter': 'standard'
        },
    },
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s %(module)s: %(message)s'
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO'
    },
    'loggers': {
        'main': {
            'handlers': ['file'],
            'level': 'INFO',
        }
    }
}

Q_CLUSTER['redis'] = {
    'host': '127.0.0.1',
    'port': 6379,
    'db': 0,
    'password': None,
    'socket_timeout': None,
    'charset': 'utf-8',
    'errors': 'strict',
    'unix_socket_path': None
}
