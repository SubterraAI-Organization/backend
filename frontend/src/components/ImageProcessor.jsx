import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  Button,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Container,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardMedia,
  CardContent,
} from "@mui/material";
import ModelSelector from "./ModelSelector";
import DatasetCreator from "./DatasetCreator";
import ImageUploader from "./ImageUploader";
import { useMaskData } from "../context/MaskDataContext";
import { TextField } from "@mui/material";

const steps = ["Select Model", "Create Dataset", "Upload Images", "Process Images"];

function ImageProcessor({ onProcessingComplete }) {
  const { setMaskData } = useMaskData();
  const [activeStep, setActiveStep] = useState(0);
  const [selectedModel, setSelectedModel] = useState(null);
  const [currentDataset, setCurrentDataset] = useState(null);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);
  const [threshold, setThreshold] = useState(20);

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
    setError(null);
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
    setError(null);
  };

  const handleDatasetCreated = (dataset) => {
    setCurrentDataset(dataset);
    setActiveStep(prev => prev + 1); // Auto-advance after creation
  };

  // Move TextField into the form JSX and fix the threshold update
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return <ModelSelector selectedModel={selectedModel} setSelectedModel={setSelectedModel} />;
      case 1:
        return <DatasetCreator onDatasetCreated={handleDatasetCreated} />;
      case 2:
        return (
          <Box>
            <TextField
              label="Area Threshold (%)"
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(Math.min(100, Math.max(0, e.target.value)))}
              inputProps={{ min: 0, max: 100 }}
              sx={{ mb: 2 }}
            />
            <ImageUploader images={images} setImages={setImages} />
          </Box>
        );
      case 3:
        return (
          <Box sx={{ mt: 3 }}>
            {results.length > 0 ? (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Processing Results
                </Typography>
                <Grid container spacing={2}>
                  {results.map((result, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                      <Card>
                        <CardMedia
                          component="img"
                          height="200"
                          image={result.maskData.image}
                          alt={`Processed image ${index + 1}`}
                          sx={{ objectFit: 'contain', bgcolor: '#f5f5f5' }}
                        />
                        <CardContent>
                          <Typography variant="body2" gutterBottom>
                            Root Count: {result.maskData.root_count}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            Avg. Diameter: {result.maskData.average_root_diameter.toFixed(2)} mm
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            Total Length: {result.maskData.total_root_length.toFixed(2)} mm
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            ) : (
              <Typography>
                Click "Process Images" to start processing your uploaded images.
              </Typography>
            )}
          </Box>
        );
      default:
        return "Unknown step";
    }
  };

  // Update the processing call with proper image IDs
  const handleProcessImages = async () => {
    try {
      setLoading(true);
      const selectedIds = images.map(img => img.id).join(',');
      
      const response = await axios.post(
        `${import.meta.env.VITE_BACKEND_URL}/api/datasets/${currentDataset.id}/images/predict/?ids=${selectedIds}`,
        {
          threshold: threshold,
          model_type: selectedModel
        },
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
          },
        }
      );

      // Handle response properly
      setResults(response.data);
      setMaskData(response.data); // Update context
      onProcessingComplete?.(response.data);
      
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Paper sx={{ p: 3, my: 3 }}>
        <Typography variant="h4" align="center" gutterBottom>
          Root Image Processor
        </Typography>
        
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {getStepContent(activeStep)}
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
          <Button
            variant="contained"
            disabled={activeStep === 0 || loading}
            onClick={handleBack}
          >
            Back
          </Button>
          
          {activeStep === steps.length - 1 ? (
            <Button
              variant="contained"
              color="primary"
              onClick={handleProcessImages}
              disabled={loading || images.length === 0}
            >
              {loading ? <CircularProgress size={24} /> : "Process Images"}
            </Button>
          ) : (
            <Button
              variant="contained"
              color="primary"
              onClick={handleNext}
              disabled={
                (activeStep === 0 && !selectedModel) ||
                loading
              }
            >
              Next
            </Button>
          )}
        </Box>
      </Paper>
    </Container>
  );
}

export default ImageProcessor;
