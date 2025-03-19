import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Box,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
  FormControlLabel,
  Switch,
  CircularProgress,
  IconButton,
  Alert,
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";
import AddIcon from "@mui/icons-material/Add";

function ModelManagement() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openAddDialog, setOpenAddDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [currentModel, setCurrentModel] = useState(null);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    public: true,
    file: null,
  });

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await axios.get(
        `${import.meta.env.VITE_BACKEND_URL}/api/models/`,
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
          },
        }
      );
      setModels(response.data);
    } catch (error) {
      console.error("Error fetching models:", error);
      setError("Failed to load models. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, checked, type, files } = e.target;
    
    if (type === "file") {
      setFormData({
        ...formData,
        file: files[0],
      });
    } else if (type === "checkbox") {
      setFormData({
        ...formData,
        [name]: checked,
      });
    } else {
      setFormData({
        ...formData,
        [name]: value,
      });
    }
  };

  const handleAddModel = async () => {
    if (!formData.name || !formData.file) {
      setError("Name and model file are required");
      return;
    }

    setLoading(true);
    try {
      const modelFormData = new FormData();
      modelFormData.append("name", formData.name);
      modelFormData.append("description", formData.description);
      modelFormData.append("public", formData.public);
      modelFormData.append("file", formData.file);

      await axios.post(
        `${import.meta.env.VITE_BACKEND_URL}/api/models/`,
        modelFormData,
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      // Reset form and close dialog
      setFormData({
        name: "",
        description: "",
        public: true,
        file: null,
      });
      setOpenAddDialog(false);
      
      // Refresh models list
      fetchModels();
    } catch (error) {
      console.error("Error adding model:", error);
      setError("Failed to add model. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleEditModel = async () => {
    if (!formData.name || !currentModel) {
      setError("Name is required");
      return;
    }

    setLoading(true);
    try {
      const modelFormData = new FormData();
      modelFormData.append("name", formData.name);
      modelFormData.append("description", formData.description);
      modelFormData.append("public", formData.public);
      
      if (formData.file) {
        modelFormData.append("file", formData.file);
      }

      await axios.patch(
        `${import.meta.env.VITE_BACKEND_URL}/api/models/${currentModel.id}/`,
        modelFormData,
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      // Reset form and close dialog
      setFormData({
        name: "",
        description: "",
        public: true,
        file: null,
      });
      setCurrentModel(null);
      setOpenEditDialog(false);
      
      // Refresh models list
      fetchModels();
    } catch (error) {
      console.error("Error updating model:", error);
      setError("Failed to update model. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteModel = async () => {
    if (!currentModel) return;

    setLoading(true);
    try {
      await axios.delete(
        `${import.meta.env.VITE_BACKEND_URL}/api/models/${currentModel.id}/`,
        {
          headers: {
            Authorization: import.meta.env.VITE_AUTHORIZATION,
            accept: "application/json",
            "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
          },
        }
      );

      setCurrentModel(null);
      setOpenDeleteDialog(false);
      
      // Refresh models list
      fetchModels();
    } catch (error) {
      console.error("Error deleting model:", error);
      setError("Failed to delete model. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const openEdit = (model) => {
    setCurrentModel(model);
    setFormData({
      name: model.name,
      description: model.description || "",
      public: model.public,
      file: null,
    });
    setOpenEditDialog(true);
  };

  const openDelete = (model) => {
    setCurrentModel(model);
    setOpenDeleteDialog(true);
  };

  return (
    <Box sx={{ my: 3 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Typography variant="h5">Model Management</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenAddDialog(true)}
        >
          Add New Model
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {loading && !openAddDialog && !openEditDialog && !openDeleteDialog ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Public</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    No models found. Add a new model to get started.
                  </TableCell>
                </TableRow>
              ) : (
                models.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell>{model.name}</TableCell>
                    <TableCell>{model.description || "No description"}</TableCell>
                    <TableCell>{model.public ? "Yes" : "No"}</TableCell>
                    <TableCell>{new Date(model.created_at).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <IconButton color="primary" onClick={() => openEdit(model)}>
                        <EditIcon />
                      </IconButton>
                      <IconButton color="error" onClick={() => openDelete(model)}>
                        <DeleteIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Add Model Dialog */}
      <Dialog open={openAddDialog} onClose={() => setOpenAddDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Model</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Upload a new model file and provide details.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            name="name"
            label="Model Name"
            type="text"
            fullWidth
            variant="outlined"
            value={formData.name}
            onChange={handleInputChange}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            name="description"
            label="Description"
            type="text"
            fullWidth
            variant="outlined"
            multiline
            rows={3}
            value={formData.description}
            onChange={handleInputChange}
            sx={{ mb: 2 }}
          />
          <FormControlLabel
            control={
              <Switch
                checked={formData.public}
                onChange={handleInputChange}
                name="public"
              />
            }
            label="Public Model"
            sx={{ mb: 2 }}
          />
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Model File
            </Typography>
            <input
              type="file"
              accept=".h5,.hdf5,.pb,.onnx,.pt,.pth"
              onChange={handleInputChange}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenAddDialog(false)} disabled={loading}>
            Cancel
          </Button>
          <Button onClick={handleAddModel} disabled={loading}>
            {loading ? <CircularProgress size={24} /> : "Add Model"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Model Dialog */}
      <Dialog open={openEditDialog} onClose={() => setOpenEditDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Model</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            name="name"
            label="Model Name"
            type="text"
            fullWidth
            variant="outlined"
            value={formData.name}
            onChange={handleInputChange}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            name="description"
            label="Description"
            type="text"
            fullWidth
            variant="outlined"
            multiline
            rows={3}
            value={formData.description}
            onChange={handleInputChange}
            sx={{ mb: 2 }}
          />
          <FormControlLabel
            control={
              <Switch
                checked={formData.public}
                onChange={handleInputChange}
                name="public"
              />
            }
            label="Public Model"
            sx={{ mb: 2 }}
          />
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Model File (Optional)
            </Typography>
            <Typography variant="caption" display="block" gutterBottom>
              Only upload a new file if you want to replace the existing model
            </Typography>
            <input
              type="file"
              accept=".h5,.hdf5,.pb,.onnx,.pt,.pth"
              onChange={handleInputChange}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenEditDialog(false)} disabled={loading}>
            Cancel
          </Button>
          <Button onClick={handleEditModel} disabled={loading}>
            {loading ? <CircularProgress size={24} /> : "Update Model"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={openDeleteDialog} onClose={() => setOpenDeleteDialog(false)}>
        <DialogTitle>Delete Model</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the model "{currentModel?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDeleteDialog(false)} disabled={loading}>
            Cancel
          </Button>
          <Button onClick={handleDeleteModel} color="error" disabled={loading}>
            {loading ? <CircularProgress size={24} /> : "Delete"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default ModelManagement;