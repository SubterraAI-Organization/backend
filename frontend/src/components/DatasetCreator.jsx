import React, { useState, forwardRef, useImperativeHandle } from "react";
import axios from "axios";
import {
    TextField,
    Button,
    Box,
    Typography,
    CircularProgress,
} from "@mui/material";

const DatasetCreator = forwardRef(
    (
        {
            onDatasetCreated,
            hideButton = false,
            datasetName: externalDatasetName,
            onDatasetNameChange,
        },
        ref
    ) => {
        const [datasetName, setDatasetName] = useState(
            externalDatasetName || ""
        );
        const [loading, setLoading] = useState(false);
        const [error, setError] = useState(null);

        // Update internal state when external prop changes
        React.useEffect(() => {
            if (externalDatasetName !== undefined) {
                setDatasetName(externalDatasetName);
            }
        }, [externalDatasetName]);

        const handleDatasetNameChange = (newName) => {
            setDatasetName(newName);
            if (onDatasetNameChange) {
                onDatasetNameChange(newName);
            }
        };

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
                if (onDatasetNameChange) {
                    onDatasetNameChange("");
                }
            } catch (error) {
                console.error("Error creating dataset:", error);
                setError("Failed to create dataset. Please try again.");
            } finally {
                setLoading(false);
            }
        };

        // Expose the create function to parent component
        useImperativeHandle(ref, () => ({
            createDataset: handleCreateDataset,
            canCreate: datasetName.trim().length > 0 && !loading,
        }));

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
                        onChange={(e) =>
                            handleDatasetNameChange(e.target.value)
                        }
                        fullWidth
                        error={!!error}
                        helperText={error}
                        disabled={loading}
                        onKeyPress={(e) => {
                            if (e.key === "Enter" && !hideButton) {
                                handleCreateDataset();
                            }
                        }}
                    />
                    {!hideButton && (
                        <Button
                            variant="contained"
                            onClick={handleCreateDataset}
                            disabled={loading}
                            sx={{ height: 56 }}
                        >
                            {loading ? (
                                <CircularProgress size={24} />
                            ) : (
                                "Create"
                            )}
                        </Button>
                    )}
                </Box>
            </Box>
        );
    }
);

DatasetCreator.displayName = "DatasetCreator";

export default DatasetCreator;
