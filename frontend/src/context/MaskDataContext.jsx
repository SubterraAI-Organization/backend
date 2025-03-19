import React, { createContext, useState, useContext } from "react";

const MaskDataContext = createContext();

export function MaskDataProvider({ children }) {
  const [maskData, setMaskData] = useState(null);

  return (
    <MaskDataContext.Provider value={{ maskData, setMaskData }}>
      {children}
    </MaskDataContext.Provider>
  );
}

export function useMaskData() {
  const context = useContext(MaskDataContext);
  if (!context) {
    throw new Error("useMaskData must be used within a MaskDataProvider");
  }
  return context;
}
