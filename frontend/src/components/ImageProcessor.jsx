import React, { useState, useEffect, useRef } from "react";
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
    TextField,
    Slider,
    FormControl,
    InputLabel,
    Input,
    InputAdornment,
} from "@mui/material";
import ModelSelector from "./ModelSelector";
import DatasetCreator from "./DatasetCreator";
import ImageUploader from "./ImageUploader";
import RefinementToggle from "./RefinementToggle";
import MultiImageViewer from "./MultiImageViewer";
import { useMaskData } from "../context/MaskDataContext";
import { imageApi, maskApi, handleApiError } from "../utils/api";
import ResultsDisplay from "./ResultsDisplay";

const steps = [
    "Select Model",
    "Create Dataset",
    "Upload Images",
    "Process Images",
];

// Default confidence thresholds by model type
const DEFAULT_CONFIDENCE_THRESHOLDS = {
    yolo: 0.3,
    unet: 0.7,
};

function ImageProcessor({ onProcessingComplete }) {
    const { setMaskData } = useMaskData();
    const [activeStep, setActiveStep] = useState(0);
    const [selectedModel, setSelectedModel] = useState(null);
    const [currentDataset, setCurrentDataset] = useState(null);
    const [datasetName, setDatasetName] = useState("");
    const [images, setImages] = useState([]);
    const [uploadedImages, setUploadedImages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [results, setResults] = useState([]);
    const [areaThreshold, setAreaThreshold] = useState(15);
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.3);
    const [isUploading, setIsUploading] = useState(false);
    const [useRefinement, setUseRefinement] = useState(false);
    const [refinementMethod, setRefinementMethod] = useState("additive");
    const [processingStatus, setProcessingStatus] = useState(null); // null, 'predicting', 'refining'

    // Add ref for DatasetCreator
    const datasetCreatorRef = useRef();

    useEffect(() => {
        // Set model-specific default confidence thresholds
        if (selectedModel) {
            if (selectedModel.toLowerCase().includes("yolo")) {
                setConfidenceThreshold(0.3);
            } else {
                setConfidenceThreshold(0.7);
            }
        }
    }, [selectedModel]);

    const handleModelChange = (model) => {
        setSelectedModel(model);
        setError(null);
    };

    const handleBack = () => {
        setActiveStep((prevStep) => prevStep - 1);
        setError(null);
    };

    const handleNext = async () => {
        // If on dataset creation step, trigger dataset creation
        if (activeStep === 1) {
            if (datasetCreatorRef.current?.canCreate) {
                try {
                    await datasetCreatorRef.current.createDataset();
                    // Success will be handled by handleDatasetCreated which advances the step
                } catch (error) {
                    setError("Failed to create dataset. Please try again.");
                }
                return;
            } else {
                setError("Please enter a dataset name");
                return;
            }
        }

        // If going from image upload step to processing step, upload images first
        if (activeStep === 2) {
            const unuploadedImages = images.filter((img) => !img.id);
            if (unuploadedImages.length > 0) {
                await uploadImages(unuploadedImages);
            }
        }

        setActiveStep((prevStep) => prevStep + 1);
        setError(null);
    };

    const uploadImages = async (imagesToUpload) => {
        if (!currentDataset || !imagesToUpload.length) return;

        setIsUploading(true);
        setError(null);

        try {
            const uploadedImageData = [];
            const newImages = [...images];

            // Upload each image
            for (let i = 0; i < imagesToUpload.length; i++) {
                const img = imagesToUpload[i];
                // Update the image status to uploading
                const imgIndex = newImages.findIndex(
                    (image) =>
                        image.name === img.name && image.preview === img.preview
                );
                if (imgIndex !== -1) {
                    newImages[imgIndex] = {
                        ...newImages[imgIndex],
                        uploading: true,
                    };
                    setImages(newImages);
                }

                try {
                    // Upload the image
                    const response = await imageApi.upload(
                        currentDataset.id,
                        img.file
                    );

                    // Update the image with the server response data
                    if (imgIndex !== -1) {
                        newImages[imgIndex] = {
                            ...newImages[imgIndex],
                            id: response.id,
                            uploading: false,
                            uploaded: true,
                            url: response.image,
                        };
                        uploadedImageData.push(response);
                    }
                } catch (err) {
                    if (imgIndex !== -1) {
                        newImages[imgIndex] = {
                            ...newImages[imgIndex],
                            uploading: false,
                            error: err.message || "Failed to upload",
                        };
                    }
                    console.error("Error uploading image:", err);
                }
            }

            setImages(newImages);
            setUploadedImages([...uploadedImages, ...uploadedImageData]);
        } catch (err) {
            handleApiError(err, setError);
        } finally {
            setIsUploading(false);
        }
    };

    const handleDatasetCreated = (dataset) => {
        setCurrentDataset(dataset);
        setActiveStep((prev) => prev + 1); // Auto-advance after creation
    };

    const getStepContent = (step) => {
        switch (step) {
            case 0:
                return (
                    <ModelSelector
                        selectedModel={selectedModel}
                        setSelectedModel={handleModelChange}
                    />
                );
            case 1:
                return (
                    <DatasetCreator
                        onDatasetCreated={handleDatasetCreated}
                        ref={datasetCreatorRef}
                        hideButton={true}
                        datasetName={datasetName}
                        onDatasetNameChange={setDatasetName}
                    />
                );
            case 2:
                return (
                    <Box>
                        <Paper sx={{ p: 3, mb: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Processing Parameters
                            </Typography>

                            {/* Refinement Toggle */}
                            <RefinementToggle
                                enabled={useRefinement}
                                onChange={setUseRefinement}
                                disabled={loading}
                            />

                            <Grid container spacing={3}>
                                <Grid item xs={12} sm={6}>
                                    <FormControl
                                        fullWidth
                                        sx={{ my: 1 }}
                                        variant="outlined"
                                    >
                                        <Typography
                                            id="area-threshold-label"
                                            gutterBottom
                                        >
                                            Area Threshold (%)
                                        </Typography>
                                        <Slider
                                            value={areaThreshold}
                                            onChange={(e, newValue) =>
                                                setAreaThreshold(newValue)
                                            }
                                            aria-labelledby="area-threshold-label"
                                            valueLabelDisplay="auto"
                                            step={1}
                                            min={0}
                                            max={100}
                                        />
                                        <Input
                                            value={areaThreshold}
                                            onChange={(e) =>
                                                setAreaThreshold(
                                                    Math.min(
                                                        100,
                                                        Math.max(
                                                            0,
                                                            e.target.value
                                                        )
                                                    )
                                                )
                                            }
                                            endAdornment={
                                                <InputAdornment position="end">
                                                    %
                                                </InputAdornment>
                                            }
                                            inputProps={{
                                                step: 1,
                                                min: 0,
                                                max: 100,
                                                type: "number",
                                                "aria-labelledby":
                                                    "area-threshold-label",
                                            }}
                                        />
                                    </FormControl>
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <FormControl
                                        fullWidth
                                        sx={{ my: 1 }}
                                        variant="outlined"
                                    >
                                        <Typography
                                            id="confidence-threshold-label"
                                            gutterBottom
                                        >
                                            Confidence Threshold
                                        </Typography>
                                        <Slider
                                            value={confidenceThreshold}
                                            onChange={(e, newValue) =>
                                                setConfidenceThreshold(newValue)
                                            }
                                            aria-labelledby="confidence-threshold-label"
                                            valueLabelDisplay="auto"
                                            step={0.05}
                                            min={0}
                                            max={1}
                                        />
                                        <Input
                                            value={confidenceThreshold}
                                            onChange={(e) =>
                                                setConfidenceThreshold(
                                                    Math.min(
                                                        1,
                                                        Math.max(
                                                            0,
                                                            e.target.value
                                                        )
                                                    )
                                                )
                                            }
                                            inputProps={{
                                                step: 0.05,
                                                min: 0,
                                                max: 1,
                                                type: "number",
                                                "aria-labelledby":
                                                    "confidence-threshold-label",
                                            }}
                                        />
                                    </FormControl>
                                </Grid>
                            </Grid>
                        </Paper>
                        <ImageUploader images={images} setImages={setImages} />
                        {isUploading && (
                            <Box
                                sx={{
                                    display: "flex",
                                    justifyContent: "center",
                                    mt: 2,
                                }}
                            >
                                <CircularProgress size={24} sx={{ mr: 1 }} />
                                <Typography>Uploading images...</Typography>
                            </Box>
                        )}
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
                                        <Grid
                                            item
                                            xs={12}
                                            sm={6}
                                            md={4}
                                            key={index}
                                        >
                                            <Card>
                                                <CardMedia
                                                    component="img"
                                                    height="200"
                                                    image={result.imageUrl}
                                                    alt={`Processed image ${
                                                        index + 1
                                                    }`}
                                                    sx={{
                                                        objectFit: "contain",
                                                        bgcolor: "#f5f5f5",
                                                    }}
                                                />
                                                <CardContent>
                                                    <Typography
                                                        variant="body2"
                                                        gutterBottom
                                                    >
                                                        Root Count:{" "}
                                                        {
                                                            result.maskData
                                                                .root_count
                                                        }
                                                    </Typography>
                                                    <Typography
                                                        variant="body2"
                                                        gutterBottom
                                                    >
                                                        Avg. Diameter:{" "}
                                                        {result.maskData.average_root_diameter.toFixed(
                                                            2
                                                        )}{" "}
                                                        mm
                                                    </Typography>
                                                    <Typography
                                                        variant="body2"
                                                        gutterBottom
                                                    >
                                                        Total Length:{" "}
                                                        {result.maskData.total_root_length.toFixed(
                                                            2
                                                        )}{" "}
                                                        mm
                                                    </Typography>
                                                </CardContent>
                                            </Card>
                                        </Grid>
                                    ))}
                                </Grid>
                            </Box>
                        ) : (
                            <Typography>
                                Click "Process Images" to start processing your
                                uploaded images.
                            </Typography>
                        )}
                    </Box>
                );
            default:
                return "Unknown step";
        }
    };

    const handleProcessImages = async () => {
        try {
            setLoading(true);
            setError(null);
            setProcessingStatus("predicting");

            // Get IDs of uploaded images
            const uploadedImageIds = images
                .filter((img) => img.id)
                .map((img) => img.id);

            if (!uploadedImageIds.length) {
                setError(
                    "No uploaded images to process. Please upload images first."
                );
                setLoading(false);
                setProcessingStatus(null);
                return;
            }

            // Make sure we have a valid model type
            if (!selectedModel) {
                setError("Please select a model type first.");
                setLoading(false);
                setProcessingStatus(null);
                return;
            }

            console.log(
                `Processing images with model type: ${selectedModel}, confidence threshold: ${confidenceThreshold}, area threshold: ${areaThreshold}, refinement: ${useRefinement}`
            );
            console.log(`Image IDs to process: ${uploadedImageIds.join(",")}`);

            try {
                // Use the enhanced bulk prediction API
                const options = {
                    modelType: selectedModel.toLowerCase(),
                    confidenceThreshold: confidenceThreshold,
                    areaThreshold: areaThreshold,
                    useRefinement: useRefinement,
                    refinementMethod: refinementMethod,
                };

                console.log("Prediction options:", options);

                // Update status if refinement is enabled
                if (useRefinement) {
                    setProcessingStatus("predicting");
                    setTimeout(() => {
                        if (loading) setProcessingStatus("refining");
                    }, 3000); // Show refining status after initial prediction
                }

                const response = await maskApi.bulkPredict(
                    currentDataset.id,
                    uploadedImageIds,
                    options
                );

                console.log("Prediction API response:", response);

                // Start the process to fetch masks
                fetchMasks(uploadedImageIds.join(","));
            } catch (apiError) {
                console.error("Prediction API error:", apiError);
                console.log("Error response:", apiError.response?.data);

                // Show a more detailed error message to the user
                let errorMsg = "Failed to start image processing.";
                if (apiError.response?.data?.detail) {
                    errorMsg += ` Server says: ${apiError.response.data.detail}`;
                }
                setError(errorMsg);
                setLoading(false);
                setProcessingStatus(null);
            }
        } catch (error) {
            console.error("Image processing error:", error);
            setError(error.response?.data?.detail || error.message);
            setLoading(false);
            setProcessingStatus(null);
        }
    };

    const fetchMasks = async (uploadedImageIds) => {
        try {
            // First wait 3 seconds before starting to check
            console.log("Waiting for processing to begin...");
            await new Promise((resolve) => setTimeout(resolve, 3000));

            // Then poll every 5 seconds for up to 25 seconds (5 attempts)
            const maxAttempts = 5; // Increase max attempts to 25 seconds total
            let attempt = 0;

            const pollForMasks = async () => {
                attempt++;
                console.log(
                    `Polling for masks, attempt ${attempt}/${maxAttempts}`
                );

                if (!uploadedImageIds) {
                    console.error("No image IDs provided to fetchMasks");
                    setError(
                        "An error occurred: No image IDs available for checking results"
                    );
                    setLoading(false);
                    return;
                }

                const imageIds = uploadedImageIds.split(",");
                console.log(
                    `Checking for masks for image IDs: ${imageIds.join(", ")}`
                );

                try {
                    // Get original images to pair with masks
                    const originalImages = await axios.get(
                        `${import.meta.env.VITE_BACKEND_URL}/api/datasets/${
                            currentDataset.id
                        }/images/`,
                        {
                            headers: {
                                Authorization: import.meta.env
                                    .VITE_AUTHORIZATION,
                                accept: "application/json",
                                "X-CSRFTOKEN": import.meta.env.VITE_CSRFTOKEN,
                            },
                        }
                    );

                    // Map image IDs to their data
                    const imageMap = {};
                    originalImages.data.forEach((img) => {
                        imageMap[img.id] = img;
                    });

                    // For each image, try to fetch its mask
                    const processedImages = [];
                    let maskFound = false;
                    let errorMessages = [];

                    for (const imageId of imageIds) {
                        try {
                            console.log(
                                `Fetching mask for image ID: ${imageId}`
                            );
                            const maskResponse = await axios.get(
                                `${
                                    import.meta.env.VITE_BACKEND_URL
                                }/api/datasets/${
                                    currentDataset.id
                                }/images/${imageId}/masks/`,
                                {
                                    headers: {
                                        Authorization: import.meta.env
                                            .VITE_AUTHORIZATION,
                                        accept: "application/json",
                                        "X-CSRFTOKEN": import.meta.env
                                            .VITE_CSRFTOKEN,
                                    },
                                    timeout: 10000, // 10 second timeout
                                }
                            );

                            console.log(
                                `Mask response for image ${imageId}:`,
                                maskResponse.data
                            );

                            // If mask exists, add it to processed images
                            if (
                                maskResponse.data &&
                                maskResponse.data.length > 0
                            ) {
                                maskFound = true;
                                const mask = maskResponse.data[0];
                                processedImages.push({
                                    imageUrl: imageMap[imageId]?.image || "",
                                    maskData: mask,
                                });
                            } else {
                                console.log(
                                    `No mask data found for image ${imageId}`
                                );
                            }
                        } catch (err) {
                            const errorMsg = `Failed to fetch mask for image ${imageId}: ${err.message}`;
                            console.log(errorMsg);
                            errorMessages.push(errorMsg);
                        }
                    }

                    // If we found at least one mask, or if we've exhausted all attempts
                    if (maskFound || attempt >= maxAttempts) {
                        if (processedImages.length > 0) {
                            console.log(
                                `Found ${processedImages.length} of ${imageIds.length} processed images`
                            );
                            setResults(processedImages);
                            setMaskData(
                                processedImages.length > 0
                                    ? {
                                          ...processedImages[0].maskData,
                                          imageUrl:
                                              processedImages[0].maskData
                                                  ?.image,
                                      }
                                    : null
                            );
                            onProcessingComplete?.(processedImages);

                            if (processedImages.length < imageIds.length) {
                                setError(
                                    `Note: Only ${processedImages.length} of ${imageIds.length} images were processed successfully.`
                                );
                            }
                        } else {
                            // If we tried all attempts and still no masks
                            if (attempt >= maxAttempts) {
                                console.error(
                                    "No masks found after maximum attempts"
                                );

                                // Check if the worker is running by making a status request
                                try {
                                    console.log(
                                        "Checking overall API status..."
                                    );
                                    const statusResponse = await axios.get(
                                        `${
                                            import.meta.env.VITE_BACKEND_URL
                                        }/api/datasets/${currentDataset.id}`,
                                        {
                                            headers: {
                                                Authorization: import.meta.env
                                                    .VITE_AUTHORIZATION,
                                                accept: "application/json",
                                                "X-CSRFTOKEN": import.meta.env
                                                    .VITE_CSRFTOKEN,
                                            },
                                        }
                                    );

                                    // If we can reach the API but no masks are generated, the worker might be stuck
                                    const errorDetails =
                                        errorMessages.length > 0
                                            ? `\n\nDiagnostic info: ${errorMessages.join(
                                                  "; "
                                              )}`
                                            : "";

                                    setError(
                                        `Image processing failed. The background worker might not be processing tasks correctly. Please check server logs.${errorDetails}`
                                    );
                                } catch (statusErr) {
                                    // If we can't even reach the API, there might be network issues
                                    setError(
                                        "Failed to communicate with the server. Please check your network connection and try again."
                                    );
                                }
                            } else {
                                // Continue polling
                                setTimeout(pollForMasks, 5000); // Wait 5 seconds before trying again
                                return; // Don't set loading to false yet
                            }
                        }

                        setLoading(false);
                        setProcessingStatus(null);
                    } else {
                        // Continue polling
                        setTimeout(pollForMasks, 5000); // Wait 5 seconds before trying again
                    }
                } catch (fetchErr) {
                    console.error("Error fetching image data:", fetchErr);
                    if (attempt >= maxAttempts) {
                        setError(
                            `Failed to fetch image data. Server responded with: ${fetchErr.message}`
                        );
                        setLoading(false);
                        setProcessingStatus(null);
                    } else {
                        // Continue polling despite error
                        setTimeout(pollForMasks, 5000);
                    }
                }
            };

            // Start polling
            await pollForMasks();
        } catch (err) {
            console.error("Error fetching masks:", err);
            setError(
                "Failed to fetch processed images. Please try again later."
            );
            setLoading(false);
            setProcessingStatus(null);
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

                <Box
                    sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mt: 3,
                    }}
                >
                    <Button
                        variant="contained"
                        disabled={activeStep === 0 || loading || isUploading}
                        onClick={handleBack}
                    >
                        Back
                    </Button>

                    {activeStep === steps.length - 1 ? (
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={handleProcessImages}
                            disabled={
                                loading || images.length === 0 || isUploading
                            }
                        >
                            {loading ? (
                                <Box
                                    sx={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 1,
                                    }}
                                >
                                    <CircularProgress size={20} />
                                    <span>
                                        {processingStatus === "predicting" &&
                                            "Predicting..."}
                                        {processingStatus === "refining" &&
                                            "Refining..."}
                                        {!processingStatus && "Processing..."}
                                    </span>
                                </Box>
                            ) : (
                                "Process Images"
                            )}
                        </Button>
                    ) : (
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={handleNext}
                            disabled={
                                (activeStep === 0 && !selectedModel) ||
                                (activeStep === 1 && !datasetName.trim()) ||
                                (activeStep === 2 && images.length === 0) ||
                                loading ||
                                isUploading
                            }
                        >
                            {isUploading ? (
                                <CircularProgress size={24} />
                            ) : activeStep === 1 ? (
                                "Create Dataset"
                            ) : (
                                "Next"
                            )}
                        </Button>
                    )}
                </Box>
            </Paper>
        </Container>
    );
}

export default ImageProcessor;
