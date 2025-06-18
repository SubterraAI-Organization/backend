import axios from "axios";

const API_URL = import.meta.env.VITE_BACKEND_URL;

// Default headers for all requests
const getHeaders = () => {
    return {
        // Remove authorization headers
        accept: "application/json",
    };
};

// API functions for datasets
export const datasetApi = {
    getAll: async () => {
        const response = await axios.get(`${API_URL}/api/datasets/`, {
            headers: getHeaders(),
        });
        return response.data;
    },

    getById: async (id) => {
        const response = await axios.get(`${API_URL}/api/datasets/${id}/`, {
            headers: getHeaders(),
        });
        return response.data;
    },

    create: async (data) => {
        const response = await axios.post(`${API_URL}/api/datasets/`, data, {
            headers: getHeaders(),
        });
        return response.data;
    },

    delete: async (id) => {
        await axios.delete(`${API_URL}/api/datasets/${id}/`, {
            headers: getHeaders(),
        });
    },
};

// API functions for images
export const imageApi = {
    getByDataset: async (datasetId) => {
        const response = await axios.get(
            `${API_URL}/api/datasets/${datasetId}/images/`,
            {
                headers: getHeaders(),
            }
        );
        return response.data;
    },

    upload: async (datasetId, imageFile) => {
        const formData = new FormData();
        formData.append("image", imageFile);

        const response = await axios.post(
            `${API_URL}/api/datasets/${datasetId}/images/`,
            formData,
            {
                headers: {
                    ...getHeaders(),
                    "Content-Type": "multipart/form-data",
                },
            }
        );
        return response.data;
    },

    delete: async (datasetId, imageId) => {
        await axios.delete(
            `${API_URL}/api/datasets/${datasetId}/images/${imageId}/`,
            {
                headers: getHeaders(),
            }
        );
    },
};

// API functions for masks
export const maskApi = {
    getByImage: async (datasetId, imageId) => {
        const response = await axios.get(
            `${API_URL}/api/datasets/${datasetId}/images/${imageId}/masks/`,
            {
                headers: getHeaders(),
            }
        );
        return response.data;
    },

    create: async (datasetId, imageId, options = {}) => {
        const {
            modelType = "unet",
            confidenceThreshold,
            areaThreshold = 0,
            useRefinement = false,
            refinementMethod = "additive",
        } = options;

        const requestData = {
            model_type: modelType,
            area_threshold: areaThreshold,
            use_refinement: useRefinement,
            refinement_method: refinementMethod,
        };

        if (confidenceThreshold !== undefined) {
            requestData.confidence_threshold = confidenceThreshold;
        }

        const response = await axios.post(
            `${API_URL}/api/datasets/${datasetId}/images/${imageId}/masks/`,
            requestData,
            {
                headers: getHeaders(),
            }
        );
        return response.data;
    },

    bulkPredict: async (datasetId, imageIds, options = {}) => {
        const {
            modelType = "unet",
            confidenceThreshold,
            areaThreshold = 0,
            useRefinement = false,
            refinementMethod = "additive",
        } = options;

        const requestData = {
            ids: Array.isArray(imageIds) ? imageIds.join(",") : imageIds,
            model_type: modelType,
            area_threshold: areaThreshold,
            use_refinement: useRefinement,
            refinement_method: refinementMethod,
        };

        if (confidenceThreshold !== undefined) {
            requestData.confidence_threshold = confidenceThreshold;
        }

        const response = await axios.post(
            `${API_URL}/api/datasets/${datasetId}/images/predict/`,
            requestData,
            {
                headers: getHeaders(),
            }
        );
        return response.data;
    },
};

// API functions for models
export const modelApi = {
    getAll: async () => {
        const response = await axios.get(`${API_URL}/api/models/`, {
            headers: getHeaders(),
        });
        return response.data;
    },

    getById: async (id) => {
        const response = await axios.get(`${API_URL}/api/models/${id}/`, {
            headers: getHeaders(),
        });
        return response.data;
    },

    create: async (formData) => {
        const response = await axios.post(`${API_URL}/api/models/`, formData, {
            headers: {
                ...getHeaders(),
                "Content-Type": "multipart/form-data",
            },
        });
        return response.data;
    },

    update: async (id, formData) => {
        const response = await axios.patch(
            `${API_URL}/api/models/${id}/`,
            formData,
            {
                headers: {
                    ...getHeaders(),
                    "Content-Type": "multipart/form-data",
                },
            }
        );
        return response.data;
    },

    delete: async (id) => {
        await axios.delete(`${API_URL}/api/models/${id}/`, {
            headers: getHeaders(),
        });
    },
};

// Helper function to handle API errors
export const handleApiError = (error, setErrorFunction) => {
    console.error("API Error:", error);

    if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        const errorMessage =
            error.response.data.detail ||
            error.response.data.message ||
            `Error: ${error.response.status} ${error.response.statusText}`;
        setErrorFunction(errorMessage);
    } else if (error.request) {
        // The request was made but no response was received
        setErrorFunction(
            "No response received from server. Please check your connection."
        );
    } else {
        // Something happened in setting up the request that triggered an Error
        setErrorFunction(`Error: ${error.message}`);
    }
};
