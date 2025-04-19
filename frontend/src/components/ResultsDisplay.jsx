import React, { useState, useEffect } from "react";
import {
    Box,
    Typography,
    Grid,
    Card,
    CardMedia,
    CardContent,
    Tabs,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Button,
    Divider,
    Chip,
    IconButton,
    Dialog,
    DialogContent,
    DialogTitle,
    Alert,
    CircularProgress,
} from "@mui/material";
import DownloadIcon from "@mui/icons-material/Download";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import CompareIcon from "@mui/icons-material/Compare";
import CloseIcon from "@mui/icons-material/Close";
import { useMaskData } from "../context/MaskDataContext";

function ResultsDisplay({ results = [] }) {
    const { maskData, setMaskData } = useMaskData();
    const [selectedTab, setSelectedTab] = useState(0);
    const [viewType, setViewType] = useState("grid"); // grid or detail
    const [openDialog, setOpenDialog] = useState(false);
    const [dialogImage, setDialogImage] = useState({ url: "", title: "" });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Validate the results when they change
    useEffect(() => {
        if (results && results.length > 0) {
            // Check if any results have incomplete data
            const invalidResults = results.filter(
                (result) =>
                    !result.imageUrl ||
                    !result.maskData ||
                    !result.maskData.image
            );

            if (invalidResults.length > 0) {
                console.warn(
                    "Some results have incomplete data:",
                    invalidResults
                );
                setError(
                    "Some results have incomplete data and may not display correctly."
                );
            } else {
                setError(null);
            }

            // Update the maskData context with the first valid result
            if (results[selectedTab] && results[selectedTab].maskData) {
                setMaskData({
                    ...results[selectedTab].maskData,
                    imageUrl: results[selectedTab].maskData.image,
                });
            }
        }
    }, [results, selectedTab, setMaskData]);

    const handleTabChange = (event, newValue) => {
        setSelectedTab(newValue);
        if (
            results &&
            results.length > 0 &&
            results[newValue] &&
            results[newValue].maskData
        ) {
            setMaskData({
                ...results[newValue].maskData,
                imageUrl: results[newValue].maskData.image,
            });
        }
    };

    const handleViewChange = (type) => {
        setViewType(type);
    };

    const handleOpenDialog = (imageUrl, title) => {
        if (!imageUrl) {
            console.warn("Attempted to open dialog with empty image URL");
            return;
        }
        setDialogImage({ url: imageUrl, title });
        setOpenDialog(true);
    };

    const handleCloseDialog = () => {
        setOpenDialog(false);
    };

    const handleDownloadCSV = () => {
        if (!results || !results.length) return;

        // Create CSV content
        const headers = [
            "Image Name",
            "Root Count",
            "Average Root Diameter (mm)",
            "Total Root Length (mm)",
            "Total Root Area (mm²)",
            "Total Root Volume (mm³)",
        ];

        const rows = results.map((result, index) => [
            `Image ${index + 1}`,
            result.maskData?.root_count || 0,
            result.maskData?.average_root_diameter?.toFixed(2) || 0,
            result.maskData?.total_root_length?.toFixed(2) || 0,
            result.maskData?.total_root_area?.toFixed(2) || 0,
            result.maskData?.total_root_volume?.toFixed(2) || 0,
        ]);

        const csvContent = [
            headers.join(","),
            ...rows.map((row) => row.join(",")),
        ].join("\n");

        // Create and download the file
        const blob = new Blob([csvContent], {
            type: "text/csv;charset=utf-8;",
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", "root_analysis_results.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    if (!results.length) {
        return (
            <Box sx={{ my: 3, textAlign: "center" }}>
                <Typography variant="h6">No results available</Typography>
                <Typography variant="body2" color="textSecondary">
                    Process some images to see results here
                </Typography>
            </Box>
        );
    }

    // Check if we have error
    if (error) {
        return (
            <Box sx={{ my: 3 }}>
                <Alert severity="warning" sx={{ mb: 2 }}>
                    {error}
                </Alert>
                {results.length > 0 && (
                    <Typography variant="body2" color="textSecondary">
                        Showing partial results that could be loaded
                    </Typography>
                )}
                {/* Continue to render available results below */}
            </Box>
        );
    }

    return (
        <Box sx={{ my: 3 }}>
            <Box
                sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    mb: 2,
                }}
            >
                <Typography variant="h5">Analysis Results</Typography>
                <Box>
                    <Button
                        variant="contained"
                        startIcon={<DownloadIcon />}
                        onClick={handleDownloadCSV}
                        sx={{ mr: 1 }}
                    >
                        Download CSV
                    </Button>
                    <Button
                        variant={viewType === "grid" ? "contained" : "outlined"}
                        onClick={() => handleViewChange("grid")}
                        sx={{ mr: 1 }}
                    >
                        Grid View
                    </Button>
                    <Button
                        variant={
                            viewType === "detail" ? "contained" : "outlined"
                        }
                        onClick={() => handleViewChange("detail")}
                    >
                        Detail View
                    </Button>
                </Box>
            </Box>

            <Divider sx={{ mb: 3 }} />

            {viewType === "grid" ? (
                <Grid container spacing={3}>
                    {results.map((result, index) => (
                        <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                            <Card
                                sx={{
                                    height: "100%",
                                    display: "flex",
                                    flexDirection: "column",
                                }}
                            >
                                <Box sx={{ position: "relative" }}>
                                    {result.imageUrl ? (
                                        <CardMedia
                                            component="img"
                                            height="180"
                                            image={result.imageUrl}
                                            alt={`Original image ${index + 1}`}
                                            sx={{ objectFit: "cover" }}
                                        />
                                    ) : (
                                        <Box
                                            sx={{
                                                height: 180,
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                                bgcolor: "#f5f5f5",
                                            }}
                                        >
                                            <Typography color="text.secondary">
                                                Image not available
                                            </Typography>
                                        </Box>
                                    )}
                                    <Box
                                        sx={{
                                            position: "absolute",
                                            top: 0,
                                            right: 0,
                                            zIndex: 1,
                                            p: 0.5,
                                        }}
                                    >
                                        {result.imageUrl && (
                                            <IconButton
                                                size="small"
                                                sx={{
                                                    bgcolor:
                                                        "rgba(255,255,255,0.7)",
                                                    mr: 1,
                                                }}
                                                onClick={() =>
                                                    handleOpenDialog(
                                                        result.imageUrl,
                                                        "Original Image"
                                                    )
                                                }
                                            >
                                                <ZoomInIcon />
                                            </IconButton>
                                        )}
                                        {result.maskData?.image && (
                                            <IconButton
                                                size="small"
                                                sx={{
                                                    bgcolor:
                                                        "rgba(255,255,255,0.7)",
                                                }}
                                                onClick={() =>
                                                    handleOpenDialog(
                                                        result.maskData.image,
                                                        "Processed Mask"
                                                    )
                                                }
                                            >
                                                <CompareIcon />
                                            </IconButton>
                                        )}
                                    </Box>
                                </Box>
                                <CardContent sx={{ flexGrow: 1 }}>
                                    <Typography
                                        gutterBottom
                                        variant="h6"
                                        component="div"
                                    >
                                        Image {index + 1}
                                    </Typography>
                                    <Box sx={{ mb: 1 }}>
                                        <Chip
                                            label={`${
                                                result.maskData?.root_count || 0
                                            } roots`}
                                            size="small"
                                            color="primary"
                                            sx={{ mb: 0.5, mr: 0.5 }}
                                        />
                                        <Chip
                                            label={`${(
                                                result.maskData
                                                    ?.total_root_length || 0
                                            ).toFixed(1)} mm length`}
                                            size="small"
                                            sx={{ mb: 0.5 }}
                                        />
                                    </Box>
                                    <Typography
                                        variant="body2"
                                        color="text.secondary"
                                    >
                                        Avg. Diameter:{" "}
                                        {(
                                            result.maskData
                                                ?.average_root_diameter || 0
                                        ).toFixed(2)}{" "}
                                        mm
                                    </Typography>
                                    <Typography
                                        variant="body2"
                                        color="text.secondary"
                                    >
                                        Area:{" "}
                                        {(
                                            result.maskData?.total_root_area ||
                                            0
                                        ).toFixed(2)}{" "}
                                        mm²
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            ) : (
                <>
                    <Box
                        sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}
                    >
                        <Tabs
                            value={selectedTab}
                            onChange={handleTabChange}
                            variant="scrollable"
                            scrollButtons="auto"
                        >
                            {results.map((_, index) => (
                                <Tab key={index} label={`Image ${index + 1}`} />
                            ))}
                        </Tabs>
                    </Box>

                    {results.map((result, index) => (
                        <Box
                            key={index}
                            sx={{
                                display:
                                    selectedTab === index ? "block" : "none",
                            }}
                        >
                            <Grid container spacing={3}>
                                <Grid item xs={12} md={6}>
                                    <Card>
                                        <Box sx={{ position: "relative" }}>
                                            {result.imageUrl ? (
                                                <CardMedia
                                                    component="img"
                                                    height="300"
                                                    image={result.imageUrl}
                                                    alt={`Original image ${
                                                        index + 1
                                                    }`}
                                                    sx={{
                                                        objectFit: "contain",
                                                        bgcolor: "#f5f5f5",
                                                    }}
                                                />
                                            ) : (
                                                <Box
                                                    sx={{
                                                        height: 300,
                                                        display: "flex",
                                                        alignItems: "center",
                                                        justifyContent:
                                                            "center",
                                                        bgcolor: "#f5f5f5",
                                                    }}
                                                >
                                                    <Typography color="text.secondary">
                                                        Image not available
                                                    </Typography>
                                                </Box>
                                            )}
                                            {result.imageUrl && (
                                                <IconButton
                                                    sx={{
                                                        position: "absolute",
                                                        top: 8,
                                                        right: 8,
                                                        bgcolor:
                                                            "rgba(255,255,255,0.7)",
                                                    }}
                                                    onClick={() =>
                                                        handleOpenDialog(
                                                            result.imageUrl,
                                                            "Original Image"
                                                        )
                                                    }
                                                >
                                                    <ZoomInIcon />
                                                </IconButton>
                                            )}
                                        </Box>
                                        <CardContent>
                                            <Typography
                                                variant="h6"
                                                align="center"
                                            >
                                                Original Image
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                                <Grid item xs={12} md={6}>
                                    <Card>
                                        <Box sx={{ position: "relative" }}>
                                            {result.maskData?.image ? (
                                                <CardMedia
                                                    component="img"
                                                    height="300"
                                                    image={
                                                        result.maskData.image
                                                    }
                                                    alt={`Processed mask ${
                                                        index + 1
                                                    }`}
                                                    sx={{
                                                        objectFit: "contain",
                                                        bgcolor: "#f5f5f5",
                                                    }}
                                                />
                                            ) : (
                                                <Box
                                                    sx={{
                                                        height: 300,
                                                        display: "flex",
                                                        alignItems: "center",
                                                        justifyContent:
                                                            "center",
                                                        bgcolor: "#f5f5f5",
                                                    }}
                                                >
                                                    <Typography color="text.secondary">
                                                        Mask not available
                                                    </Typography>
                                                </Box>
                                            )}
                                            {result.maskData?.image && (
                                                <IconButton
                                                    sx={{
                                                        position: "absolute",
                                                        top: 8,
                                                        right: 8,
                                                        bgcolor:
                                                            "rgba(255,255,255,0.7)",
                                                    }}
                                                    onClick={() =>
                                                        handleOpenDialog(
                                                            result.maskData
                                                                .image,
                                                            "Processed Mask"
                                                        )
                                                    }
                                                >
                                                    <ZoomInIcon />
                                                </IconButton>
                                            )}
                                        </Box>
                                        <CardContent>
                                            <Typography
                                                variant="h6"
                                                align="center"
                                            >
                                                Processed Mask
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                                <Grid item xs={12}>
                                    <TableContainer component={Paper}>
                                        <Table>
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>
                                                        Metric
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        Value
                                                    </TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    <TableCell>
                                                        Root Count
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {result.maskData
                                                            ?.root_count || 0}
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>
                                                        Average Root Diameter
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {(
                                                            result.maskData
                                                                ?.average_root_diameter ||
                                                            0
                                                        ).toFixed(2)}{" "}
                                                        mm
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>
                                                        Total Root Length
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {(
                                                            result.maskData
                                                                ?.total_root_length ||
                                                            0
                                                        ).toFixed(2)}{" "}
                                                        mm
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>
                                                        Total Root Area
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {(
                                                            result.maskData
                                                                ?.total_root_area ||
                                                            0
                                                        ).toFixed(2)}{" "}
                                                        mm²
                                                    </TableCell>
                                                </TableRow>
                                                <TableRow>
                                                    <TableCell>
                                                        Total Root Volume
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {(
                                                            result.maskData
                                                                ?.total_root_volume ||
                                                            0
                                                        ).toFixed(2)}{" "}
                                                        mm³
                                                    </TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Grid>
                            </Grid>
                        </Box>
                    ))}
                </>
            )}

            {/* Image Dialog */}
            <Dialog
                open={openDialog}
                onClose={handleCloseDialog}
                maxWidth="lg"
                fullWidth
            >
                <DialogTitle>
                    <Box
                        sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                        }}
                    >
                        {dialogImage.title}
                        <IconButton
                            edge="end"
                            color="inherit"
                            onClick={handleCloseDialog}
                            aria-label="close"
                        >
                            <CloseIcon />
                        </IconButton>
                    </Box>
                </DialogTitle>
                <DialogContent>
                    <img
                        src={dialogImage.url}
                        alt={dialogImage.title}
                        style={{
                            width: "100%",
                            height: "auto",
                            maxHeight: "80vh",
                            objectFit: "contain",
                        }}
                    />
                </DialogContent>
            </Dialog>
        </Box>
    );
}

export default ResultsDisplay;
