import React, { useState, useEffect } from "react";
import {
    Box,
    Typography,
    FormControl,
    RadioGroup,
    FormControlLabel,
    Radio,
    Paper,
    CircularProgress,
    Alert,
} from "@mui/material";
import { modelApi, handleApiError } from "../utils/api";

function ModelSelector({ selectedModel, setSelectedModel }) {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const data = await modelApi.getAll();
            setModels(data);

            // If no model is selected and we have models, select the first one's model_type
            if (!selectedModel && data.length > 0) {
                setSelectedModel(data[0].model_type);
            }
        } catch (error) {
            handleApiError(error, setError);
        } finally {
            setLoading(false);
        }
    };

    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
    };

    // If no custom models are available, show built-in model options
    const getDefaultModels = () => {
        return [
            {
                id: 1,
                name: "UNet",
                model_type: "unet",
                description:
                    "Best for detailed segmentation with high accuracy",
            },
            {
                id: 2,
                name: "YOLO",
                model_type: "yolo",
                description: "Fast real-time object detection and segmentation",
            },
        ];
    };

    const displayModels = models.length > 0 ? models : getDefaultModels();

    // Make sure a model is selected after loading models
    useEffect(() => {
        if (!loading && !selectedModel && displayModels.length > 0) {
            setSelectedModel(displayModels[0].model_type.toLowerCase());
        }
    }, [loading, displayModels, selectedModel, setSelectedModel]);

    if (loading) {
        return (
            <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box>
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
                <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                        Select AI Model
                    </Typography>
                    <FormControl component="fieldset">
                        <RadioGroup
                            aria-label="model"
                            name="model"
                            value={selectedModel || ""}
                            onChange={handleModelChange}
                        >
                            {getDefaultModels().map((model) => (
                                <FormControlLabel
                                    key={model.id}
                                    value={model.model_type}
                                    control={<Radio />}
                                    label={
                                        <Box>
                                            <Typography variant="body1">
                                                {model.name}
                                            </Typography>
                                            {model.description && (
                                                <Typography
                                                    variant="caption"
                                                    color="textSecondary"
                                                >
                                                    {model.description}
                                                </Typography>
                                            )}
                                        </Box>
                                    }
                                />
                            ))}
                        </RadioGroup>
                    </FormControl>
                </Paper>
            </Box>
        );
    }

    return (
        <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
                Select AI Model
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
                Choose the AI model to process your root images.
            </Typography>

            <FormControl component="fieldset">
                <RadioGroup
                    aria-label="model"
                    name="model"
                    value={selectedModel || ""}
                    onChange={handleModelChange}
                >
                    {displayModels.map((model) => (
                        <FormControlLabel
                            key={model.id}
                            value={model.model_type}
                            control={<Radio />}
                            label={
                                <Box>
                                    <Typography variant="body1">
                                        {model.name}
                                    </Typography>
                                    {model.description && (
                                        <Typography
                                            variant="caption"
                                            color="textSecondary"
                                        >
                                            {model.description}
                                        </Typography>
                                    )}
                                </Box>
                            }
                        />
                    ))}
                </RadioGroup>
            </FormControl>
        </Paper>
    );
}

export default ModelSelector;
