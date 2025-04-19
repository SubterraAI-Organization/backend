import React, { useState, useRef } from "react";
import {
    Box,
    Button,
    Typography,
    Grid,
    Card,
    CardMedia,
    CardContent,
    IconButton,
    CircularProgress,
    Tooltip,
    Badge,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DeleteIcon from "@mui/icons-material/Delete";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";

function ImageUploader({ images, setImages }) {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    };

    const handleFiles = (files) => {
        const newImages = Array.from(files).map((file) => ({
            file,
            preview: URL.createObjectURL(file),
            name: file.name,
            uploading: false,
        }));

        setImages([...images, ...newImages]);
    };

    const removeImage = (index) => {
        const newImages = [...images];
        // Revoke the object URL to avoid memory leaks
        URL.revokeObjectURL(newImages[index].preview);
        newImages.splice(index, 1);
        setImages(newImages);
    };

    return (
        <Box sx={{ my: 3 }}>
            <Typography variant="h6" gutterBottom>
                Upload Images
            </Typography>

            <Box
                sx={{
                    border: `2px dashed ${isDragging ? "#1976d2" : "#aaa"}`,
                    borderRadius: 2,
                    p: 3,
                    textAlign: "center",
                    backgroundColor: isDragging
                        ? "rgba(25, 118, 210, 0.04)"
                        : "transparent",
                    transition: "all 0.3s",
                    cursor: "pointer",
                    mb: 3,
                }}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current.click()}
            >
                <input
                    type="file"
                    multiple
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    style={{ display: "none" }}
                    accept="image/*"
                />
                <CloudUploadIcon
                    sx={{
                        fontSize: 48,
                        color: isDragging ? "#1976d2" : "#aaa",
                        mb: 1,
                    }}
                />
                <Typography variant="h6">Drag & Drop Images Here</Typography>
                <Typography variant="body2" color="textSecondary">
                    or click to browse files
                </Typography>
            </Box>

            {images.length > 0 && (
                <Grid container spacing={2}>
                    {images.map((image, index) => (
                        <Grid item xs={6} sm={4} md={3} key={index}>
                            <Card>
                                <Box sx={{ position: "relative" }}>
                                    <CardMedia
                                        component="img"
                                        height="140"
                                        image={image.preview || image.url}
                                        alt={image.name}
                                        sx={{
                                            objectFit: "contain",
                                            bgcolor: "#f5f5f5",
                                        }}
                                    />
                                    {image.uploaded && (
                                        <Box
                                            sx={{
                                                position: "absolute",
                                                top: 5,
                                                right: 5,
                                            }}
                                        >
                                            <Tooltip title="Uploaded successfully">
                                                <CheckCircleIcon color="success" />
                                            </Tooltip>
                                        </Box>
                                    )}
                                    {image.error && (
                                        <Box
                                            sx={{
                                                position: "absolute",
                                                top: 5,
                                                right: 5,
                                            }}
                                        >
                                            <Tooltip title={image.error}>
                                                <ErrorIcon color="error" />
                                            </Tooltip>
                                        </Box>
                                    )}
                                </Box>
                                <CardContent
                                    sx={{ p: 1, "&:last-child": { pb: 1 } }}
                                >
                                    <Box
                                        sx={{
                                            display: "flex",
                                            justifyContent: "space-between",
                                            alignItems: "center",
                                        }}
                                    >
                                        <Tooltip title={image.name}>
                                            <Typography
                                                variant="body2"
                                                noWrap
                                                sx={{ maxWidth: "70%" }}
                                            >
                                                {image.name}
                                            </Typography>
                                        </Tooltip>
                                        {image.uploading ? (
                                            <CircularProgress size={24} />
                                        ) : (
                                            <IconButton
                                                size="small"
                                                onClick={() =>
                                                    removeImage(index)
                                                }
                                            >
                                                <DeleteIcon fontSize="small" />
                                            </IconButton>
                                        )}
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            )}
        </Box>
    );
}

export default ImageUploader;
