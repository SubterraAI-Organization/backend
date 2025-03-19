import React, { useState } from "react";
import axios from "axios";
import { TextField, Button, Box, Typography, CircularProgress } from "@mui/material";

function DatasetCreator({ onDatasetCreated }) {
  const [datasetName, setDatasetName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCreateDataset = async () => {
    if (!datasetName.trim()) {
      setError("Please enter a dataset name");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `${import.meta.env.VITE_BACKEND_URL}/api/datasets/`,
        { name: datasetName },
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
          },
        }
      );

      if (onDatasetCreated) {
        onDatasetCreated(response.data);
      }
      
      setDatasetName("");
    } catch (error) {
      console.error("Error creating dataset:", error);
      setError("Failed to create dataset. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ my: 3 }}>
      <Typography variant="h6" gutterBottom>
        Create New Dataset
      </Typography>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <TextField
          label="Dataset Name"
          variant="outlined"
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
          fullWidth
          error={!!error}
          helperText={error}
          disabled={loading}
        />
        <Button
          variant="contained"
          onClick={handleCreateDataset}
          disabled={loading}
          sx={{ height: 56 }}
        >
          {loading ? <CircularProgress size={24} /> : "Create"}
        </Button>
      </Box>
    </Box>
  );
}

export default DatasetCreator;