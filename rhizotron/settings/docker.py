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
