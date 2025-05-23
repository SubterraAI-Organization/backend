import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  Box,
  Typography,
  Grid,
  Card,
  CardMedia,
  CardContent,
  CardActions,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert,
  IconButton,
} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import VisibilityIcon from "@mui/icons-material/Visibility";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";

function DatasetDetail({ datasetId, onBack }) {
  const [dataset, setDataset] = useState(null);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [imageMasks, setImageMasks] = useState({}); // { imageId: maskData }

  useEffect(() => {
    fetchDatasetDetails();
  }, [datasetId]);

  const fetchDatasetDetails = async () => {
    setLoading(true);
    try {
      // Fetch dataset details
      const datasetResponse = await axios.get(
        `${import.meta.env.VITE_BACKEND_URL}/api/datasets/${datasetId}/`,
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
          },
        }
      );
      setDataset(datasetResponse.data);

      // Fetch images in the dataset
      const imagesResponse = await axios.get(
        `${import.meta.env.VITE_BACKEND_URL}/api/datasets/${datasetId}/images/`,
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
          },
        }
      );
      setImages(imagesResponse.data);
    } catch (error) {
      console.error("Error fetching dataset details:", error);
      setError("Failed to load dataset details. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (images.length > 0) {
      const fetchAllMasks = async () => {
        const maskResults = {};
        await Promise.all(
          images.map(async (img) => {
            try {
              const res = await axios.get(
                `${import.meta.env.VITE_BACKEND_URL}/api/datasets/${datasetId}/images/${img.id}/masks/`,
                {
                  headers: {
                    Authorization: import.meta.env.VITE_AUTHORIZATION,
                    accept: "application/json",
                    "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
                  },
                }
              );
              // Only store the first mask (if any)
              maskResults[img.id] = res.data && res.data.length > 0 ? res.data[0] : null;
            } catch (err) {
              maskResults[img.id] = null;
            }
          })
        );
        setImageMasks(maskResults);
      };
      fetchAllMasks();
    }
  }, [images, datasetId]);

  const handleImageClick = (image) => {
    setSelectedImage(image);
    setOpenDialog(true);
  };

  const handleDeleteImage = async () => {
    if (!selectedImage) return;

    setDeleteLoading(true);
    try {
      await axios.delete(
        `${import.meta.env.VITE_BACKEND_URL}/api/datasets/${datasetId}/images/${selectedImage.id}/`,
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
          },
        }
      );

      // Remove the deleted image from the state
      setImages(images.filter(img => img.id !== selectedImage.id));
      setOpenDeleteDialog(false);
      setOpenDialog(false);
    } catch (error) {
      console.error("Error deleting image:", error);
      setError("Failed to delete image. Please try again.");
    } finally {
      setDeleteLoading(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ my: 3 }}>
        <Button startIcon={<ArrowBackIcon />} onClick={onBack} sx={{ mb: 2 }}>
          Back to Datasets
        </Button>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!dataset) {
    return (
      <Box sx={{ my: 3 }}>
        <Button startIcon={<ArrowBackIcon />} onClick={onBack} sx={{ mb: 2 }}>
          Back to Datasets
        </Button>
        <Typography>Dataset not found.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ my: 3 }}>
      <Button startIcon={<ArrowBackIcon />} onClick={onBack} sx={{ mb: 2 }}>
        Back to Datasets
      </Button>
      
      <Typography variant="h5" gutterBottom>
        Dataset: {dataset.name}
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="body2" color="textSecondary">
          Created: {new Date(dataset.created_at).toLocaleString()}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Status: {dataset.public ? "Public" : "Private"}
        </Typography>
      </Box>

      <Typography variant="h6" gutterBottom>
        Images ({images.length})
      </Typography>

      {images.length === 0 ? (
        <Typography>No images in this dataset.</Typography>
      ) : (
        <Grid container spacing={2}>
          {images.map((image) => (
            <Grid item xs={6} sm={4} md={3} key={image.id}>
              <Card>
                <CardMedia
                  component="img"
                  height="160"
                  image={image.image}
                  alt={`Image ${image.id}`}
                  sx={{ objectFit: "contain", bgcolor: "#f5f5f5" }}
                />
                <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                  <Typography variant="body2" noWrap>
                    Image {image.id}
                  </Typography>
                  {imageMasks[image.id] && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="caption">Root Count: {imageMasks[image.id].root_count}</Typography><br />
                      <Typography variant="caption">Avg Diameter: {imageMasks[image.id].average_root_diameter?.toFixed(2)}</Typography><br />
                      <Typography variant="caption">Total Length: {imageMasks[image.id].total_root_length?.toFixed(2)}</Typography><br />
                      <Typography variant="caption">Avg Length: {imageMasks[image.id].average_root_length?.toFixed(2)}</Typography><br />
                      <Typography variant="caption">Area: {imageMasks[image.id].total_root_area?.toFixed(2)}</Typography><br />
                      <Typography variant="caption">Volume: {imageMasks[image.id].total_root_volume?.toFixed(2)}</Typography>
                    </Box>
                  )}
                </CardContent>
                <CardActions>
                  <IconButton size="small" onClick={() => handleImageClick(image)}>
                    <VisibilityIcon fontSize="small" />
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Image Detail Dialog */}
      <Dialog
        open={openDialog}
        onClose={() => setOpenDialog(false)}
        maxWidth="md"
        fullWidth
      >
        {selectedImage && (
          <>
            <DialogTitle>
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <Typography variant="h6">Image Details</Typography>
                <IconButton color="error" onClick={() => setOpenDeleteDialog(true)}>
                  <DeleteIcon />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Box sx={{ textAlign: "center", mb: 2 }}>
                <img
                  src={selectedImage.image}
                  alt={`Image ${selectedImage.id}`}
                  style={{ maxWidth: "100%", maxHeight: "60vh" }}
                />
              </Box>
              <Typography variant="body1" gutterBottom>
                Image ID: {selectedImage.id}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Uploaded: {new Date(selectedImage.created_at).toLocaleString()}
              </Typography>
              {selectedImage && imageMasks[selectedImage.id] && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2">Metrics</Typography>
                  <Typography variant="body2">Root Count: {imageMasks[selectedImage.id].root_count}</Typography>
                  <Typography variant="body2">Average Root Diameter: {imageMasks[selectedImage.id].average_root_diameter?.toFixed(2)}</Typography>
                  <Typography variant="body2">Total Root Length: {imageMasks[selectedImage.id].total_root_length?.toFixed(2)}</Typography>
                  <Typography variant="body2">Average Root Length: {imageMasks[selectedImage.id].average_root_length?.toFixed(2)}</Typography>
                  <Typography variant="body2">Total Root Area: {imageMasks[selectedImage.id].total_root_area?.toFixed(2)}</Typography>
                  <Typography variant="body2">Total Root Volume: {imageMasks[selectedImage.id].total_root_volume?.toFixed(2)}</Typography>
                </Box>
              )}
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setOpenDialog(false)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={openDeleteDialog} onClose={() => setOpenDeleteDialog(false)}>
        <DialogTitle>Delete Image</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this image? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDeleteDialog(false)} disabled={deleteLoading}>
            Cancel
          </Button>
          <Button
            onClick={handleDeleteImage}
            color="error"
            disabled={deleteLoading}
          >
            {deleteLoading ? <CircularProgress size={24} /> : "Delete"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default DatasetDetail;