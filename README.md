# RhizoRoot.ai

RhizoRoot.ai is a web application that allows users to segment and analyze root images. The application is built using Django, React, and PyTorch. The application is designed to be user-friendly and easy to use. The application allows users to upload root images, segment the roots, and analyze the root images. The application also allows users to train their own models using their own data. The application is designed to be easy to use and user-friendly. The application is designed to be used by researchers, scientists, and students who are interested in root analysis.

#### Table of Contents

| [Usage](#usage) | [Directory Structure](#Directory-Structure) | [License](#license) | [Contact](#contact) |

## Usage

Clear Docker Cache
```sh
docker system prune -a
```

To run locally (if anything is new)
```sh
docker compose up --build -d
```

To use RhizoRoot.ai, you can run the following command:

```sh
git clone https://github.com/C4theBomb/rhizotron.git
cd rhizotron
docker compose -f docker-compose.prod.yml up -d --pull always
```

To update an existing implementation, you can run the following command:

```sh
git pull
docker compose -f docker-compose.prod.yml up -d --pull always
```

## Directory Structure

```
├── rhizotron/                      # Main project directory
│   ├── settings                        # Django settings
│   └── urls.py                         # Django URL configuration
├── processing/                     # Processing and REST API implementation
│   ├── migrations/                     # Database migration files
│   ├── tests/                          # API application tests
│   ├── admin.py                        # Django admin configuration
│   ├── apps.py                         # Django app configuration
│   ├── models.py                       # Database models
│   ├── permissions.py                  # Custom API permissions
│   ├── routers.py                      # API router implementations
│   ├── serializers.py                  # API serializers
│   ├── services.py                     # API worker asynchronous services
│   ├── urls.py                         # API URL configuration
│   └── views.py                        # API view definitions
├── segmentation/                   # Contains all the model implementation
│   ├── data/                           # Data processing modules for datasets
│   ├── management/commands/            # Management commands for model creation
│   ├── models/                         # NN model implementations
│   ├── tests/                      # Contains all the tests
│   └── utils/                      # Contains all the utility functions
├── frontend/                        # Contains all the frontend implementation
│   ├── src/                            # Contains all the sourtce code
│   ├── Dockerfile                      # Frontend Dockerfile
│   └── Dockerfile.dev                  # Frontend development Dockerfile
├── logs/                           # Contains all the script logs
├── media/                          # Contains all the media files  (images, masks, etc.)
├── django.rules                    # Prometheus metrics for django exports
├── docker-compose.prod.yml         # Application docker compose for production
├── docker-compose.yml              # Application docker compose for development
├── Dockerfile                      # Server and worker Dockerfile
├── Dockerfile.dev                  # Server and worker development Dockerfile
├── prometheus.yml                  # Prometheus configuration
└── setup.cfg                       # LSP configuration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

# rootphe
