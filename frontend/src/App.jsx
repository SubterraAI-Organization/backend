import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { MaskDataProvider } from "./context/MaskDataContext.jsx";
import Footer from "./components/Footer";
import NavBar from "./components/NavBar";
import GenerateMask from "./components/GenerateMask";
import Dashboard from "./components/Dashboard";
import theme from './theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <MaskDataProvider>
          <NavBar />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/legacy" element={<GenerateMask />} />
            {/* Other routes can be added here */}
          </Routes>
          <Footer />
        </MaskDataProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;
