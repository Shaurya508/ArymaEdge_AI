import { useMemo, useState, useEffect, useRef } from "react";
import "./App.css";
import SpendSlider from "./components/SpendSlider.jsx";
import AskEmmmyDialog from "./components/AskEmmmyDialog.jsx";
import WarningTooltip from "./components/WarningTooltip.jsx";
import SaturationCurves from "./pages/SaturationCurves.jsx";
import { CHANNELS } from "./data/channels.js";
import { API_BASE_URL } from "./config.js";

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const initialValues = CHANNELS.reduce((acc, channel) => {
  acc[channel.id] = channel.id === "Base" ? 1 : 0;
  return acc;
}, {});

function App() {
  const [spendValues, setSpendValues] = useState(initialValues);
  const [resultMeta, setResultMeta] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [livePredictedSales, setLivePredictedSales] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [report, setReport] = useState(null);
  const [dialogState, setDialogState] = useState({
    open: false,
    phase: "target",
    targetSales: "",
    optimizerType: "default",
    error: "",
  });
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [salesRange, setSalesRange] = useState(null);
  const [isLoadingSalesRange, setIsLoadingSalesRange] = useState(false);
  const [isApplyingOptimization, setIsApplyingOptimization] = useState(false);
  const predictTimeoutRef = useRef(null);

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", isDarkMode ? "dark" : "light");
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode((prev) => !prev);
  };

  const seasonalityBadges = useMemo(() => {
    if (!resultMeta?.seasonality) return [];
    return Object.entries(resultMeta.seasonality)
      .filter(([, value]) => Number(value) === 1)
      .map(([label]) => label);
  }, [resultMeta]);

  const handleSliderChange = (id, value) => {
    setSpendValues((prev) => ({
      ...prev,
      [id]: value,
    }));
  };

  // Real-time prediction when sliders change
  useEffect(() => {
    if (predictTimeoutRef.current) {
      clearTimeout(predictTimeoutRef.current);
    }

    const hasNonBaseChanges = Object.entries(spendValues).some(
      ([key, val]) => key !== "Base" && val > 0
    );

    if (!hasNonBaseChanges && spendValues.Base === 1) {
      setLivePredictedSales(null);
      return;
    }

    setIsPredicting(true);
    predictTimeoutRef.current = setTimeout(async () => {
      try {
        const spendsForAPI = { ...spendValues };
        delete spendsForAPI.Base;

        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            spends: spendsForAPI,
            prediction_month: "Jan-24",
          }),
        });

        if (response.ok) {
          const data = await response.json();
          setLivePredictedSales(data.predicted_sales);
        }
      } catch (error) {
        console.error("Error predicting sales:", error);
      } finally {
        setIsPredicting(false);
      }
    }, 500);

    return () => {
      if (predictTimeoutRef.current) {
        clearTimeout(predictTimeoutRef.current);
      }
    };
  }, [spendValues]);

  const openDialog = async () => {
    setDialogState({
      open: true,
      phase: "target",
      targetSales: "",
      optimizerType: "default",
      error: "",
    });
    
    // Fetch sales range
    setIsLoadingSalesRange(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sales-range`);
      if (response.ok) {
        const data = await response.json();
        setSalesRange(data);
      }
    } catch (error) {
      console.error("Error fetching sales range:", error);
    } finally {
      setIsLoadingSalesRange(false);
    }
  };

  const closeDialog = () => {
    if (isOptimizing) return;
    setDialogState((prev) => ({ ...prev, open: false }));
  };

  const goToOptimizerStep = () => {
    if (!dialogState.targetSales || Number(dialogState.targetSales) <= 0) {
      setDialogState((prev) => ({
        ...prev,
        error: "Please enter a positive target sales number.",
      }));
      return;
    }
    
    // Check if target is within achievable range (if available)
    if (salesRange && salesRange.status === "ready") {
      const target = Number(dialogState.targetSales);
      const optimizerType = dialogState.optimizerType || "default";
      const rangeData = optimizerType === "roi" 
        ? salesRange.roi_optimizer 
        : salesRange.default_optimizer;
      
      if (rangeData) {
        const { min_sales, max_sales } = rangeData;
        
        if (target < min_sales || target > max_sales) {
          setDialogState((prev) => ({
            ...prev,
            error: `⚠️ Target sales is outside achievable range: ${currencyFormatter.format(min_sales)} - ${currencyFormatter.format(max_sales)}. The optimizer will return the closest achievable result.`,
          }));
          // Still allow proceeding, but with warning
        }
      }
    }
    
    setDialogState((prev) => ({
      ...prev,
      phase: "optimizer",
      error: prev.error || "", // Keep warning if exists
    }));
  };

  const runOptimization = async () => {
    try {
      setDialogState((prev) => ({ ...prev, phase: "loading", error: "" }));
      setIsOptimizing(true);
      // setStatusMessage("Running optimization...");
      setWarnings([]);
      setReport(null);

      const response = await fetch(`${API_BASE_URL}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_sales: Number(dialogState.targetSales),
          optimizer_type: dialogState.optimizerType,
          prediction_month: "Jan-24",
        }),
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || "Unable to optimize spends.");
      }

      const data = await response.json();

      // Set animation state
      setIsApplyingOptimization(true);

      // Apply new values
      setSpendValues((prev) => {
        const updated = { ...prev };
        if (data.spends) {
          Object.entries(data.spends).forEach(([key, value]) => {
            if (key in updated) {
              updated[key] = Number(value);
            }
          });
        }
        return updated;
      });

      // Turn off animation after completion
      setTimeout(() => {
        setIsApplyingOptimization(false);
      }, 1200);

      setResultMeta({
        optimizerType: dialogState.optimizerType,
        targetSales: Number(dialogState.targetSales),
        predictedSales: data.predicted_sales ?? null,
        seasonality: data.seasonality ?? {},
        spends: data.spends ?? {},
      });

      // Set warnings from API response
      if (data.warnings && data.warnings.length > 0) {
        setWarnings(data.warnings);
      } else {
        setWarnings([]);
      }

      // Auto-generate report
      try {
        const reportResponse = await fetch(`${API_BASE_URL}/generate-report`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            target_sales: Number(dialogState.targetSales),
            optimizer_type: dialogState.optimizerType,
            spends: data.spends ?? {},
            predicted_sales: data.predicted_sales ?? null,
            seasonality: data.seasonality ?? {},
            warnings: data.warnings ?? [],
          }),
        });

        if (reportResponse.ok) {
          const reportData = await reportResponse.json();
          setReport(reportData);
        }
      } catch (reportError) {
        console.error("Error generating report:", reportError);
      }

      setDialogState((prev) => ({ ...prev, open: false }));
    } catch (error) {
      setDialogState((prev) => ({
        ...prev,
        phase: "optimizer",
        error: error.message,
      }));
    } finally {
      setIsOptimizing(false);
    }
  };


  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-content">
          <div className="logo-icon">📊</div>
          <div>
            <p className="eyebrow"></p>
            <h1>Aryma Nebula</h1>
            <p className="subtitle">
            Agentic Marketing Spend Optimizer
            </p>
          </div>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {isDarkMode ? "☀️" : "🌙"}
          </button>
          <button type="button" className="ask-btn" onClick={openDialog}>
            Let's Optimize
          </button>
        </div>
      </header>

      {(resultMeta || livePredictedSales !== null) && (
        <section className="summary-card">
          {resultMeta && (
            <>
              <div>
                <p className="summary-label">Previous Month Sales</p>
                {/* hard coded value for previous month sales for now*/}
                <h3>{currencyFormatter.format(9337020.85)}</h3>
              </div>
              <div>
                <p className="summary-label">Optimizer</p>
                <h3>{resultMeta.optimizerType.toUpperCase()}</h3>
              </div>
              {seasonalityBadges.length > 0 && (
                <div>
                  <p className="summary-label">Seasonality</p>
                  <div className="badge-row">
                    {seasonalityBadges.map((badge) => (
                      <span key={badge} className="badge">
                        {badge}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
          <div>
            <div className="summary-label-row">
              {warnings.some((w) => w.channel === "Overall") && (
                <WarningTooltip warnings={warnings.filter((w) => w.channel === "Overall")}>
                  <span></span>
                </WarningTooltip>
              )}
              <p className="summary-label">
                {isPredicting ? "Calculating..." : "Target Sales"}
              </p>
            </div>
            <h3 className={livePredictedSales !== null ? "live-prediction" : ""}>
              {livePredictedSales !== null
                ? currencyFormatter.format(livePredictedSales)
                : resultMeta?.predictedSales
                ? currencyFormatter.format(resultMeta.predictedSales)
                : "—"}
            </h3>
          </div>
        </section>
      )}

      <div className="optimizer-layout">
        <section className="sliders-container">
          {CHANNELS.map((channel) => {
            // Get warnings for this specific channel
            const channelWarnings = warnings.filter(
              (warning) => warning.channel === channel.id
            );
            
            return (
              <SpendSlider
                key={channel.id}
                label={channel.label}
                max={channel.max}
                value={spendValues[channel.id]}
                step={channel.step || 1000}
                disabled={channel.disabled}
                onChange={(value) => handleSliderChange(channel.id, value)}
                isAnimating={isApplyingOptimization}
                warnings={channelWarnings}
              />
            );
          })}
        </section>

        <section className="live-saturation-curves">
          <SaturationCurves 
            optimizedSpends={resultMeta?.spends || null}
            compact={true}
          />
        </section>
      </div>


      {/* Report Section */}
      {report && (
        <section className="report-section">
          <div className="report-header">
            <h3 className="section-title">📊 Optimization Report</h3>
            <button
              type="button"
              className="close-report-btn"
              onClick={() => setReport(null)}
            >
              ✕
            </button>
          </div>
          
          {report.report.map((section, index) => (
            <div key={index} className="report-block">
              <h4 className="report-block-title">{section.title}</h4>
              
              {section.type === "table" && (
                <div className="report-table-wrapper">
                  <table className="report-table">
                    <thead>
                      <tr>
                        <th>Channel</th>
                        <th>Optimized</th>
                        <th>Max</th>
                        <th>Avg</th>
                        <th>% of Max</th>
                        <th>Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {section.data.map((row, i) => (
                        <tr key={i} className={`risk-${row.risk_level.toLowerCase()}`}>
                          <td>{row.channel}</td>
                          <td>{currencyFormatter.format(row.optimized_spend)}</td>
                          <td>{currencyFormatter.format(row.historical_max)}</td>
                          <td>{currencyFormatter.format(row.historical_mean)}</td>
                          <td>{row.pct_of_max.toFixed(1)}%</td>
                          <td>
                            <span className={`risk-badge risk-${row.risk_level.toLowerCase()}`}>
                              {row.risk_level}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              
              {section.type === "list" && (
                <ul className="report-list">
                  {section.items.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              )}
              
              {section.content && (
                <p className="report-content">{section.content}</p>
              )}
            </div>
          ))}
        </section>
      )}

      <AskEmmmyDialog
        open={dialogState.open}
        phase={dialogState.phase}
        targetSales={dialogState.targetSales}
        optimizerType={dialogState.optimizerType}
        salesRange={salesRange}
        isLoadingSalesRange={isLoadingSalesRange}
        onTargetChange={(value) =>
          setDialogState((prev) => ({ ...prev, targetSales: value, error: "" }))
        }
        onOptimizerChange={(value) =>
          setDialogState((prev) => ({ ...prev, optimizerType: value, error: "" }))
        }
        onClose={closeDialog}
        onNext={goToOptimizerStep}
        onBack={() =>
          setDialogState((prev) => ({ ...prev, phase: "target", error: "" }))
        }
        onRun={runOptimization}
        isLoading={isOptimizing}
        error={dialogState.error}
      />
    </div>
  );
}

export default App;
