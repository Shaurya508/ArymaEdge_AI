import { useState, useEffect, useCallback } from "react";
import Plot from 'react-plotly.js';
import { API_BASE_URL } from "../config.js";
import "./SaturationCurves.css";

export default function SaturationCurves({ optimizedSpends = null, compact = false }) {
  const [allCurvesData, setAllCurvesData] = useState(null);
  const [selectedChannel, setSelectedChannel] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  
  // Marker visibility toggles
  const [showMax, setShowMax] = useState(true);
  const [showAvg, setShowAvg] = useState(true);
  const [showSat, setShowSat] = useState(true);
  const [showOpt, setShowOpt] = useState(true);

  const fetchCurvesData = useCallback(async () => {
    try {
      setLoading(true);
      
      const url = optimizedSpends 
        ? `${API_BASE_URL}/saturation-curves-with-optimized`
        : `${API_BASE_URL}/saturation-curves`;
      
      const options = optimizedSpends 
        ? {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ optimized_spends: optimizedSpends })
          }
        : { method: 'GET' };
      
      const response = await fetch(url, options);
      if (!response.ok) throw new Error("Failed to fetch saturation curves");
      
      const data = await response.json();
      const plotlyFigure = JSON.parse(data.plotly_json);
      
      setAllCurvesData(plotlyFigure);
      
      // Set first channel as default
      if (plotlyFigure.data && plotlyFigure.data.length > 0) {
        const firstCurve = plotlyFigure.data.find(trace => trace.mode === 'lines');
        if (firstCurve) {
          setSelectedChannel(firstCurve.name);
          setCurrentIndex(0);
        }
      }
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [optimizedSpends]);

  useEffect(() => {
    fetchCurvesData();
  }, [fetchCurvesData]);

  const getChannelData = (channelName) => {
    if (!allCurvesData) return null;

    // Filter traces for the selected channel
    let channelTraces = allCurvesData.data.filter(trace => {
      // Include the line trace for this channel
      if (trace.name === channelName) return true;
      
      // Include marker traces that belong to this channel based on visibility toggles
      if (trace.legendgroup && trace.legendgroup.startsWith('markers_')) {
        if (trace.customdata && trace.customdata.length > 0) {
          const belongsToChannel = trace.customdata.some(cd => cd === channelName);
          if (!belongsToChannel) return false;
          
          // Apply visibility filters
          if (trace.legendgroup === 'markers_max' && !showMax) return false;
          if (trace.legendgroup === 'markers_avg' && !showAvg) return false;
          if (trace.legendgroup === 'markers_sat' && !showSat) return false;
          if (trace.legendgroup === 'markers_opt' && !showOpt) return false;
          
          return true;
        }
      }
      
      return false;
    });

    return {
      data: channelTraces,
      layout: {
        ...allCurvesData.layout,
        title: '',  // Remove title from chart
        showlegend: false,
        height: compact ? 380 : 450,
        margin: { l: 60, r: 20, t: 10, b: 50 }
      }
    };
  };

  const getChannelList = () => {
    if (!allCurvesData || !allCurvesData.data) return [];
    
    return allCurvesData.data
      .filter(trace => trace.mode === 'lines' && trace.name)
      .map(trace => trace.name);
  };

  const handlePrevChannel = () => {
    const channels = getChannelList();
    if (channels.length === 0) return;
    
    setIsTransitioning(true);
    setTimeout(() => {
      const newIndex = currentIndex === 0 ? channels.length - 1 : currentIndex - 1;
      setCurrentIndex(newIndex);
      setSelectedChannel(channels[newIndex]);
      setTimeout(() => setIsTransitioning(false), 50);
    }, 300);
  };

  const handleNextChannel = () => {
    const channels = getChannelList();
    if (channels.length === 0) return;
    
    setIsTransitioning(true);
    setTimeout(() => {
      const newIndex = currentIndex === channels.length - 1 ? 0 : currentIndex + 1;
      setCurrentIndex(newIndex);
      setSelectedChannel(channels[newIndex]);
      setTimeout(() => setIsTransitioning(false), 50);
    }, 300);
  };

  if (loading) {
    return (
      <div className={compact ? "saturation-curves-compact" : "saturation-curves-page"}>
        {!compact && (
        <div className="page-header">
          <h1>Saturation Curves</h1>
          <p className="page-subtitle">Sales due to Marketing Spends</p>
        </div>
        )}
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading saturation curves...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={compact ? "saturation-curves-compact" : "saturation-curves-page"}>
        {!compact && (
        <div className="page-header">
          <h1>Saturation Curves</h1>
          <p className="page-subtitle">Sales due to Marketing Spends</p>
        </div>
        )}
        <div className="error-state">
          <p>Error: {error}</p>
          <button onClick={fetchCurvesData}>Retry</button>
        </div>
      </div>
    );
  }

  const channels = getChannelList();
  const currentCurveData = selectedChannel ? getChannelData(selectedChannel) : null;
  const hasOptimized = optimizedSpends && Object.keys(optimizedSpends).length > 0;

  return (
    <div className={compact ? "saturation-curves-compact" : "saturation-curves-page"}>
      {!compact && (
      <div className="page-header">
        <h1>Saturation Curves</h1>
        <p className="page-subtitle">Sales due to Marketing Spends</p>
      </div>
      )}

      <div className="curves-container-with-nav">
        {/* Left Arrow */}
        <button 
          className="side-nav-arrow left" 
          onClick={handlePrevChannel}
          disabled={channels.length === 0}
          aria-label="Previous channel"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M15 18L9 12L15 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>

        {/* Curve Display with Channel Info */}
        <div className={`curve-main-display ${isTransitioning ? 'transitioning' : ''}`}>
          <div className="channel-title-bar">
            <h3>{selectedChannel}</h3>
            <span className="channel-badge">{currentIndex + 1} / {channels.length}</span>
          </div>
          
          {currentCurveData && (
            <>
              <div className="plot-wrapper">
                <Plot
                  data={currentCurveData.data}
                  layout={{
                    ...currentCurveData.layout,
                    autosize: true,
                    responsive: true,
                  }}
                  config={{
                    displayModeBar: false,  // Hide modebar completely
                    displaylogo: false,
                    responsive: true
                  }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler={true}
                />
              </div>
              
              {/* Marker Toggles at Bottom */}
              <div className="curve-controls-bottom">
                <button 
                  className={`marker-toggle-btn ${showMax ? 'active' : ''}`}
                  onClick={() => setShowMax(!showMax)}
                  title="Toggle Max Spend"
                >
                  ○ Max
                </button>
                
                <button 
                  className={`marker-toggle-btn ${showAvg ? 'active' : ''}`}
                  onClick={() => setShowAvg(!showAvg)}
                  title="Toggle Avg Spend"
                >
                  △ Avg
                </button>
                
                <button 
                  className={`marker-toggle-btn ${showSat ? 'active' : ''}`}
                  onClick={() => setShowSat(!showSat)}
                  title="Toggle Saturation Point"
                >
                  ◾ Sat
                </button>
                
                {hasOptimized && (
              <button
                    className={`marker-toggle-btn ${showOpt ? 'active' : ''}`}
                    onClick={() => setShowOpt(!showOpt)}
                    title="Toggle Optimized Spend"
                  >
                    ⭐ Opt
              </button>
            )}
          </div>
            </>
          )}
        </div>

        {/* Right Arrow */}
        <button 
          className="side-nav-arrow right" 
          onClick={handleNextChannel}
          disabled={channels.length === 0}
          aria-label="Next channel"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M9 18L15 12L9 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </div>
  );
}
