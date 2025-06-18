import React from "react";
import {
    FormControlLabel,
    Switch,
    Tooltip,
    Box,
    Typography,
    Chip,
} from "@mui/material";
import { Info } from "@mui/icons-material";

const RefinementToggle = ({
    enabled,
    onChange,
    disabled = false,
    showDescription = true,
}) => {
    return (
        <Box sx={{ mb: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                <FormControlLabel
                    control={
                        <Switch
                            checked={enabled}
                            onChange={(e) => onChange(e.target.checked)}
                            disabled={disabled}
                            color="primary"
                        />
                    }
                    label={
                        <Box
                            sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 1,
                            }}
                        >
                            <Typography variant="body2" fontWeight="medium">
                                Enable Refinement
                            </Typography>
                            {enabled && (
                                <Chip
                                    label="Higher Accuracy"
                                    size="small"
                                    color="success"
                                    variant="outlined"
                                />
                            )}
                        </Box>
                    }
                />

                <Tooltip
                    title="Refinement runs a second prediction pass on areas initially missed, improving detection accuracy but taking longer to process."
                    arrow
                >
                    <Info
                        sx={{
                            fontSize: 16,
                            color: "text.secondary",
                            cursor: "help",
                        }}
                    />
                </Tooltip>
            </Box>

            {showDescription && (
                <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ display: "block", ml: 4, mb: 1 }}
                >
                    {enabled
                        ? "Will perform an additional prediction pass on areas initially missed for improved accuracy."
                        : "Standard single-pass prediction for faster processing."}
                </Typography>
            )}

            {enabled && (
                <Box sx={{ ml: 4 }}>
                    <Typography variant="caption" color="warning.main">
                        ⚠️ Refinement mode takes approximately 2x longer to
                        process
                    </Typography>
                </Box>
            )}
        </Box>
    );
};

export default RefinementToggle;
