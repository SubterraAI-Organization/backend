from .defaults import *


DATABASES = {
    'default': {
        'ENGINE': 'django_prometheus.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

Q_CLUSTER = {
    'name': 'rhizotron',
    'workers': 2,
    'recycle': 200,
    'timeout': None,
    'label': 'Tasks Queue',
    'sync': True
}
