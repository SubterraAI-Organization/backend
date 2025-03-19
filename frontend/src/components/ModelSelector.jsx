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
      
      // If no model is selected and we have models, select the first one's MODEL TYPE
      if (!selectedModel && data.length > 0) {
        setSelectedModel(data[0].model_type);  // Changed from .name to .model_type
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

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (models.length === 0) {
    return (
      <Alert severity="info">
        No models available. Please add a model in the Models tab.
      </Alert>
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
          {models.map((model) => (
            <FormControlLabel
              key={model.id}
              value={model.model_type}  // Changed from model.name to model.model_type
              control={<Radio />}
              label={
                <Box>
                  <Typography variant="body1">{model.name}</Typography>
                  {model.description && (
                    <Typography variant="caption" color="textSecondary">
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