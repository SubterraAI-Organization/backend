import React, { useState, useEffect } from "react";
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Tabs,
  Tab,
  Button,
  CircularProgress,
} from "@mui/material";
import axios from "axios";
import ImageProcessor from "./ImageProcessor";
import ResultsDisplay from "./ResultsDisplay";
import ModelManagement from "./ModelManagement";
import DatasetDetail from "./DatasetDetail";

function Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [processingResults, setProcessingResults] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [viewingDataset, setViewingDataset] = useState(false);

  useEffect(() => {
    // Fetch datasets when component mounts
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const response = await axios.get(
        `${import.meta.env.VITE_BACKEND_URL}/api/datasets/`,
        {
          headers: {
            accept: "application/json", // Remove authorization headers
          },
        }
      );
      setDatasets(response.data);
    } catch (error) {
      console.error("Error fetching datasets:", error);
      setError("Failed to load datasets");
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    // Reset dataset view when changing tabs
    setViewingDataset(false);
  };

  const handleProcessingComplete = (results) => {
    setProcessingResults(results);
    // Switch to results tab
    setActiveTab(1);
  };

  const handleViewDataset = (dataset) => {
    setSelectedDataset(dataset);
    setViewingDataset(true);
  };

  const handleBackToDatasets = () => {
    setViewingDataset(false);
    // Refresh datasets list
    fetchDatasets();
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          RhizoRoot AI Dashboard
        </Typography>
        <Typography variant="body1" color="textSecondary" paragraph>
          Upload plant root images, process them with our AI models, and analyze the results.
        </Typography>
      </Paper>

      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab label="Process Images" />
          <Tab label="View Results" />
          <Tab label="Datasets" />
          <Tab label="Models" />
        </Tabs>
      </Box>

      {/* Tab content */}
      <Box sx={{ mt: 2 }}>
        {/* Process Images Tab */}
        <Box sx={{ display: activeTab === 0 ? "block" : "none" }}>
          <ImageProcessor onProcessingComplete={handleProcessingComplete} />
        </Box>

        {/* Results Tab */}
        <Box sx={{ display: activeTab === 1 ? "block" : "none" }}>
          <ResultsDisplay results={processingResults} />
        </Box>

        {/* Datasets Tab */}
        <Box sx={{ display: activeTab === 2 ? "block" : "none" }}>
          {viewingDataset && selectedDataset ? (
            <DatasetDetail 
              datasetId={selectedDataset.id} 
              onBack={handleBackToDatasets} 
            />
          ) : (
            <>
              <Typography variant="h5" gutterBottom>
                Your Datasets
              </Typography>
              
              {loading ? (
                <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
                  <CircularProgress />
                </Box>
              ) : error ? (
                <Typography color="error">{error}</Typography>
              ) : datasets.length === 0 ? (
                <Typography>No datasets found. Create one in the Process Images tab.</Typography>
              ) : (
                <Grid container spacing={3}>
                  {datasets.map((dataset) => (
                    <Grid item xs={12} sm={6} md={4} key={dataset.id}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="h6">{dataset.name}</Typography>
                        <Typography variant="body2" color="textSecondary">
                          Created: {new Date(dataset.created_at).toLocaleDateString()}
                        </Typography>
                        <Typography variant="body2">
                          Images: {dataset.image_count || 0}
                        </Typography>
                        <Box sx={{ mt: 2, display: "flex", justifyContent: "flex-end" }}>
                          <Button 
                            variant="outlined" 
                            size="small"
                            onClick={() => handleViewDataset(dataset)}
                          >
                            View Details
                          </Button>
                        </Box>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </>
          )}
        </Box>

        {/* Models Tab */}
        <Box sx={{ display: activeTab === 3 ? "block" : "none" }}>
          <ModelManagement />
        </Box>
      </Box>
    </Container>
  );
}

export default Dashboard;