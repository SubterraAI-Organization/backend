import React, { useState, useEffect } from "react";
import {
    Box,
    Card,
    CardContent,
    CardMedia,
    Typography,
    ToggleButton,
    ToggleButtonGroup,
    Grid,
    IconButton,
    Tooltip,
    Paper,
    Skeleton,
    Alert,
} from "@mui/material";
import {
    ViewModule,
    Image as ImageIcon,
    Layers,
    ZoomIn,
    ZoomOut,
    FullscreenExit,
    Fullscreen,
} from "@mui/icons-material";

const MultiImageViewer = ({
    originalImage,
    overlayImage,
    imageTitle = "Prediction Results",
    loading = false,
    error = null,
    onImageClick = null,
}) => {
    const [viewMode, setViewMode] = useState("overlay"); // Default to overlay view
    const [zoomLevel, setZoomLevel] = useState(100);
    const [fullscreen, setFullscreen] = useState(null); // Track which image is in fullscreen

    // Add ESC key support for fullscreen
    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === "Escape" && fullscreen) {
                event.preventDefault();
                event.stopPropagation();
                setFullscreen(null);
            }
        };

        document.addEventListener("keydown", handleKeyDown, true);
        return () => {
            document.removeEventListener("keydown", handleKeyDown, true);
        };
    }, [fullscreen]);

    const handleViewModeChange = (event, newMode) => {
        if (newMode !== null) {
            setViewMode(newMode);
            setFullscreen(null); // Exit fullscreen when changing modes
        }
    };

    const handleZoomIn = () => {
        setZoomLevel((prev) => Math.min(prev + 25, 200));
    };

    const handleZoomOut = () => {
        setZoomLevel((prev) => Math.max(prev - 25, 50));
    };

    const handleImageClick = (imageType, imageUrl) => {
        if (onImageClick) {
            onImageClick(imageType, imageUrl);
        }
        // Toggle fullscreen for the clicked image
        setFullscreen(fullscreen === imageType ? null : imageType);
    };

    const renderImage = (src, alt, label, imageType) => {
        if (!src) return null;

        const isFullscreen = fullscreen === imageType;

        return (
            <Card
                sx={{
                    height: "100%",
                    cursor: "pointer",
                    transition: "transform 0.2s",
                    "&:hover": {
                        transform: "scale(1.02)",
                        boxShadow: 3,
                    },
                    ...(isFullscreen && {
                        position: "fixed",
                        top: 0,
                        left: 0,
                        width: "100vw",
                        height: "100vh",
                        zIndex: 2000,
                        margin: 0,
                        borderRadius: 0,
                        transform: "none",
                        backgroundColor: "rgba(0,0,0,0.9)",
                    }),
                }}
                onClick={() => handleImageClick(imageType, src)}
            >
                <CardMedia
                    component="img"
                    image={src}
                    alt={alt}
                    sx={{
                        height: isFullscreen ? "calc(100vh - 120px)" : 400,
                        objectFit: "contain",
                        backgroundColor: isFullscreen
                            ? "transparent"
                            : "#f5f5f5",
                        transform: `scale(${zoomLevel / 100})`,
                        transformOrigin: "center center",
                        transition: "transform 0.2s",
                    }}
                />
                <CardContent
                    sx={{
                        p: 1.5,
                        backgroundColor: isFullscreen
                            ? "rgba(0,0,0,0.8)"
                            : "white",
                    }}
                >
                    <Box
                        sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                        }}
                    >
                        <Typography
                            variant="subtitle2"
                            fontWeight="medium"
                            color={isFullscreen ? "white" : "inherit"}
                        >
                            {label}
                        </Typography>
                        <Box
                            sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 0.5,
                            }}
                        >
                            <Tooltip title="Zoom In">
                                <IconButton
                                    size="small"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleZoomIn();
                                    }}
                                    sx={{
                                        color: isFullscreen
                                            ? "white"
                                            : "inherit",
                                    }}
                                >
                                    <ZoomIn fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Zoom Out">
                                <IconButton
                                    size="small"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleZoomOut();
                                    }}
                                    sx={{
                                        color: isFullscreen
                                            ? "white"
                                            : "inherit",
                                    }}
                                >
                                    <ZoomOut fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip
                                title={
                                    isFullscreen
                                        ? "Exit Fullscreen (ESC)"
                                        : "Fullscreen"
                                }
                            >
                                <IconButton
                                    size="small"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleImageClick(imageType, src);
                                    }}
                                    sx={{
                                        color: isFullscreen
                                            ? "white"
                                            : "inherit",
                                    }}
                                >
                                    {isFullscreen ? (
                                        <FullscreenExit fontSize="small" />
                                    ) : (
                                        <Fullscreen fontSize="small" />
                                    )}
                                </IconButton>
                            </Tooltip>
                        </Box>
                    </Box>
                    <Typography
                        variant="caption"
                        color={
                            isFullscreen
                                ? "rgba(255,255,255,0.7)"
                                : "text.secondary"
                        }
                    >
                        {zoomLevel}% zoom{" "}
                        {isFullscreen && "• Press ESC to exit"}
                    </Typography>
                </CardContent>
            </Card>
        );
    };

    const renderLoadingSkeleton = (label) => (
        <Card sx={{ height: "100%" }}>
            <Skeleton variant="rectangular" height={400} />
            <CardContent sx={{ p: 1.5 }}>
                <Typography variant="subtitle2">{label}</Typography>
                <Typography variant="caption" color="text.secondary">
                    Loading...
                </Typography>
            </CardContent>
        </Card>
    );

    if (error) {
        return (
            <Alert severity="error" sx={{ mt: 2 }}>
                {error}
            </Alert>
        );
    }

    return (
        <Paper sx={{ p: 3, mt: 2 }}>
            <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                    {imageTitle}
                </Typography>

                {/* View Mode Controls */}
                <Box
                    sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        mb: 2,
                    }}
                >
                    <ToggleButtonGroup
                        value={viewMode}
                        exclusive
                        onChange={handleViewModeChange}
                        size="small"
                    >
                        <ToggleButton value="sideBySide">
                            <ViewModule fontSize="small" sx={{ mr: 1 }} />
                            Side by Side
                        </ToggleButton>
                        <ToggleButton value="overlay">
                            <Layers fontSize="small" sx={{ mr: 1 }} />
                            Overlay (Default)
                        </ToggleButton>
                        <ToggleButton value="original">
                            <ImageIcon fontSize="small" sx={{ mr: 1 }} />
                            Original Only
                        </ToggleButton>
                    </ToggleButtonGroup>

                    <Typography variant="caption" color="text.secondary">
                        Click images for fullscreen • ESC to exit
                    </Typography>
                </Box>
            </Box>

            {/* Image Display */}
            {viewMode === "overlay" && (
                <Grid container spacing={2}>
                    <Grid item xs={12}>
                        {loading
                            ? renderLoadingSkeleton("Overlay Image")
                            : renderImage(
                                  overlayImage,
                                  "Overlay image with detected roots",
                                  "Detected Roots (Green Highlights)",
                                  "overlay"
                              )}
                    </Grid>
                </Grid>
            )}

            {viewMode === "sideBySide" && (
                <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                        {loading
                            ? renderLoadingSkeleton("Original Image")
                            : renderImage(
                                  originalImage,
                                  "Original image",
                                  "Original Image",
                                  "original"
                              )}
                    </Grid>
                    <Grid item xs={12} md={6}>
                        {loading
                            ? renderLoadingSkeleton("Overlay Image")
                            : renderImage(
                                  overlayImage,
                                  "Overlay image with detected roots",
                                  "Detected Roots (Green Highlights)",
                                  "overlay"
                              )}
                    </Grid>
                </Grid>
            )}

            {viewMode === "original" && (
                <Grid container spacing={2}>
                    <Grid item xs={12}>
                        {loading
                            ? renderLoadingSkeleton("Original Image")
                            : renderImage(
                                  originalImage,
                                  "Original image",
                                  "Original Image",
                                  "original"
                              )}
                    </Grid>
                </Grid>
            )}

            {/* Image Information */}
            <Box
                sx={{
                    mt: 2,
                    p: 2,
                    backgroundColor: "#f8f9fa",
                    borderRadius: 1,
                }}
            >
                <Typography variant="caption" color="text.secondary">
                    <strong>Overlay:</strong> Green highlights show detected
                    roots on the original image (Default View) •
                    <strong> Original:</strong> Raw input image without any
                    overlays
                </Typography>
            </Box>
        </Paper>
    );
};

export default MultiImageViewer;
