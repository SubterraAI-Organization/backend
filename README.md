# SubterraAI

SubterraAI is a highly modular software platform for automating image‐based root phenotyping using non-invasive data collection techniques. By integrating edge computing with advanced deep learning models and real-time image analysis from minirhizotron tubes, the platform enables researchers to capture, segment, and analyze root images across multiple soil depths without disturbing the natural soil environment.

Built using Django, React, and PyTorch, the web application is designed to be user-friendly and flexible. It supports intuitive annotation tools, custom model uploads, powerful command-line utilities, and robust RESTful API services—making it ideal for researchers, scientists, and students interested in root analysis.

#### Table of Contents

| [Usage](#usage) | [Directory Structure](#directory-structure) | [License](#license) | [Contact](#contact) |

---

## Flexibility and Customization

SubterraAI is designed with versatility in mind to meet diverse user needs. The platform offers:
- **Intuitive Annotation Tools**: Easily label and annotate root images with integrated tools to generate high-quality training datasets.
- **Custom Model Upload**: Seamlessly upload and manage custom deep learning models, enabling tailored analyses for specific research applications.
- **Command-Line Interface**: Access powerful command-line utilities for tasks such as training new models, fine-tuning existing ones, and running predictions.
- **RESTful API Services**: Leverage robust API endpoints for automated data processing, integration into existing workflows, and remote system operations.

---

## Key Features

### Split Computing for Efficient Data Processing
- **Edge Device Processing**: Executes a lightweight, split version of deep neural networks (DNNs) for preliminary image analysis, reducing latency and conserving battery life.
- **Cloud Server Processing**: Handles computationally intensive tasks (such as full-scale segmentation and trait analysis) using robust hardware, ensuring energy efficiency and real-time feedback.
- **YOLOv8 Architecture**: Leverages YOLOv8 for fast, real-time predictions while maintaining high segmentation accuracy even when the network is split between devices.

### Advanced Multimodal Deep Learning
SubterraAI supports a suite of state-of-the-art models, each tailored for different phenotyping tasks:
- **U-Net**: Provides pixel-level segmentation ideal for detailed and consistent root mapping.
- **YOLOv8**: Enables rapid object detection and segmentation, making it optimal for real-time field applications.
- **Detectron2**: Offers robust segmentation performance under heterogeneous and challenging environmental conditions.

Performance metrics include:
- *YOLOv8*: Precision – 0.85, Recall – 0.85  
- *Detectron2*: Precision – 0.98, Recall – 0.98

### Comprehensive Data Preprocessing & Transfer Learning
- **Preprocessing Techniques**: Implements Gaussian filtering, adaptive histogram equalization, Canny edge detection, and dynamic illumination correction to mitigate image artifacts (e.g., condensation, debris, and inconsistent lighting).
- **Multispectral Imaging Integration**: Uses LED lighting at different wavelengths to enhance contrast between roots and soil.
- **Transfer Learning Framework**: A base U-Net model (pre-trained on diverse datasets such as PRMI) is fine-tuned on crop-specific data (e.g., sorghum) to adapt to various field conditions while maintaining high generalization.

---

## How It Works

1. **Data Capture**  
   Minirhizotron tubes capture high-resolution images at multiple soil depths. Each image is tagged with environmental metadata (temperature, humidity, soil moisture, etc.) to enhance analysis accuracy.

2. **Edge Processing**  
   A split version of the DNN runs on edge devices for rapid preprocessing and preliminary analysis, minimizing communication overhead and conserving energy.

3. **Cloud Processing**  
   Preprocessed data is transmitted securely to cloud servers where advanced image analysis is performed using the multimodal deep learning architecture.

4. **Results Delivery**  
   Processed data—including segmented root images and extracted phenotypic traits—is returned in real time, providing actionable insights for plant breeders and researchers.

---

## Platform Architecture

SubterraAI is built with a modular, scalable architecture comprising several layers:

### 1. Presentation Layer
- **User Interface**: A ReactJS-based web application that allows users to upload images, view datasets, and interact with analysis results.
- **Visualization Tools**: Integrated dashboards and graphical tools (e.g., a Grafana-based admin panel) provide real-time insights into system performance and phenotypic data.

### 2. Business Logic Layer
- **RESTful APIs**: Exposes endpoints for managing datasets, images, masks, and models.
  - **Datasets API**: Create, retrieve, update, and delete datasets.
  - **Images API**: Upload, list, and delete images associated with datasets.
  - **Masks API**: Predict, retrieve, and export root masks (including integration with tools like LabelMe).
  - **Models API**: Manage model uploads, updates, and retrievals.
- **Data Validation & Processing**: Ensures robust image preprocessing (normalization, augmentation, and noise reduction) before analysis.

### 3. Data Access and Storage Layers
- **Data Access Layer**: Uses Object Relational Mapping (ORM) to interact with the database, streamlining CRUD operations.
- **Storage Layer**: Combines secure file storage (local or cloud-based, e.g., AWS S3) with structured database systems to manage metadata, model weights, and image files.

### Command-Line Services
The platform also provides several command-line tools to complement the web services:
- **predict.py**: Processes individual images to generate predictions.
- **decompose_layers.py**: Splits images into segments for detailed analysis.
- **build_composites.py**: Reconstructs full composite images from segmented outputs.
- **train.py**: Enables custom model training or fine-tuning using new datasets, facilitating rapid adaptation for different crops and field conditions.

---

## Performance & Evaluation

SubterraAI has been rigorously evaluated using high-performance NVIDIA H100 GPUs and benchmarks across three architectures:
- **U-Net**: Excels in detailed segmentation with high accuracy and low validation loss.
- **YOLOv8**: Offers rapid detection with steadily improving precision and recall during training.
- **Detectron2**: Provides superior segmentation performance in noisy environments, with precision stabilizing around 0.92 and recall near 0.80.

Additional evaluations include depth-wise root area comparisons across different sorghum genotypes, revealing variations in root distribution critical for drought resilience and carbon sequestration.

---

## Future Directions

SubterraAI is continuously evolving. Future enhancements include:
- **Expanded Trait Extraction**: Automated calculation of root length, diameter, branching patterns, and anatomical features.
- **Enhanced Real-Time Processing**: Further optimization for field deployment under variable environmental conditions.
- **Broader Model Integration**: Continued development of new models and transfer learning strategies to support a wider range of crops and soil types.
- **User-Centric Customization**: Improved annotation tools and command-line services for tailoring models to specific research needs.

---

## Getting Started

For installation, setup instructions, and detailed API documentation, please refer to our [GitHub repository](https://github.com/SubterraAI-Organization/backend.git).

SubterraAI is open source and welcomes contributions from the research and developer community to help advance sustainable agriculture and precision breeding.

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

Frontend UI: ```http://localhost:5173/```

Swagger API UI: ```http://localhost:8000/api/schema/swagger/#/```

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
