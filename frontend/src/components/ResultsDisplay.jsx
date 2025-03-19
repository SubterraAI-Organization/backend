import React, { useState } from "react";
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
} from "@mui/material";
import DownloadIcon from "@mui/icons-material/Download";
import { useMaskData } from "../context/MaskDataContext";

function ResultsDisplay({ results = [] }) {
  const { maskData, setMaskData } = useMaskData();
  const [selectedTab, setSelectedTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
    if (results[newValue]) {
      setMaskData({
        ...results[newValue].maskData,
        imageUrl: results[newValue].maskData.image
      });
    }
  };

  const handleDownloadCSV = () => {
    if (!results.length) return;
    
    // Create CSV content
    const headers = [
      "Image Name",
      "Root Count",
      "Average Root Diameter (mm)",
      "Total Root Length (mm)",
      "Total Root Area (mm²)",
      "Total Root Volume (mm³)"
    ];
    
    const rows = results.map((result, index) => [
      `Image ${index + 1}`,
      result.maskData.root_count,
      result.maskData.average_root_diameter.toFixed(2),
      result.maskData.total_root_length.toFixed(2),
      result.maskData.total_root_area.toFixed(2),
      result.maskData.total_root_volume.toFixed(2)
    ]);
    
    const csvContent = [
      headers.join(","),
      ...rows.map(row => row.join(","))
    ].join("\n");
    
    // Create and download the file
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
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

  return (
    <Box sx={{ my: 3 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
        <Typography variant="h5">Analysis Results</Typography>
        <Button 
          variant="contained" 
          startIcon={<DownloadIcon />}
          onClick={handleDownloadCSV}
        >
          Download CSV
        </Button>
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}>
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
        <Box key={index} sx={{ display: selectedTab === index ? "block" : "none" }}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardMedia
                  component="img"
                  height="300"
                  image={result.imageUrl}
                  alt={`Original image ${index + 1}`}
                  sx={{ objectFit: "contain", bgcolor: "#f5f5f5" }}
                />
                <CardContent>
                  <Typography variant="h6" align="center">Original Image</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardMedia
                  component="img"
                  height="300"
                  image={result.maskData.image}
                  alt={`Processed mask ${index + 1}`}
                  sx={{ objectFit: "contain", bgcolor: "#f5f5f5" }}
                />
                <CardContent>
                  <Typography variant="h6" align="center">Processed Mask</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12}>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell align="right">Value</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Root Count</TableCell>
                      <TableCell align="right">{result.maskData.root_count}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Average Root Diameter</TableCell>
                      <TableCell align="right">{result.maskData.average_root_diameter.toFixed(2)} mm</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Total Root Length</TableCell>
                      <TableCell align="right">{result.maskData.total_root_length.toFixed(2)} mm</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Total Root Area</TableCell>
                      <TableCell align="right">{result.maskData.total_root_area.toFixed(2)} mm²</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Total Root Volume</TableCell>
                      <TableCell align="right">{result.maskData.total_root_volume.toFixed(2)} mm³</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </Box>
      ))}
    </Box>
  );
}

export default ResultsDisplay;