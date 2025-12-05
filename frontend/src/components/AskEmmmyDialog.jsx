import PropTypes from "prop-types";
import "./AskEmmmyDialog.css";

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

export default function AskEmmmyDialog({
  open,
  phase,
  targetSales,
  optimizerType,
  salesRange,
  isLoadingSalesRange,
  onTargetChange,
  onOptimizerChange,
  onClose,
  onNext,
  onBack,
  onRun,
  isLoading,
  error,
}) {
  if (!open) {
    return null;
  }

  const renderBody = () => {
    if (phase === "target") {
      return (
        <>
          <p className="dialog-question">What is your target sales amount?</p>
          <input
            type="number"
            min={0}
            step={100000}
            value={targetSales}
            onChange={(e) => onTargetChange(e.target.value)}
            placeholder="e.g., 9000000"
            autoFocus
          />
          {isLoadingSalesRange && (
            <p className="sales-range-info loading">Loading achievable range...</p>
          )}
          {salesRange && salesRange.status === "ready" && (
            <div className="sales-range-info">
              <p className="range-label">Achievable Sales Range:</p>
              <p className="range-values">
                {currencyFormatter.format(salesRange.default_optimizer.min_sales)} - {currencyFormatter.format(salesRange.default_optimizer.max_sales)}
              </p>
            </div>
          )}
          {salesRange && salesRange.status === "initializing" && (
            <p className="sales-range-info warning">
              ⏳ Optimizers are initializing. You can proceed but it may take longer.
            </p>
          )}
        </>
      );
    }

    if (phase === "optimizer") {
      const showRangeForOptimizer = salesRange && salesRange.status === "ready";
      const rangeData = optimizerType === "roi" 
        ? salesRange?.roi_optimizer 
        : salesRange?.default_optimizer;
      
      return (
        <>
          <p className="dialog-question">
            Choose your optimization strategy
          </p>
          <div className="radio-group">
            <div className="radio-option">
              <input
                type="radio"
                id="opt-default"
                value="default"
                checked={optimizerType === "default"}
                onChange={(e) => onOptimizerChange(e.target.value)}
              />
              <label htmlFor="opt-default">
                <span className="icon">⚡</span>
                Default
              </label>
            </div>
            <div className="radio-option">
              <input
                type="radio"
                id="opt-roi"
                value="roi"
                checked={optimizerType === "roi"}
                onChange={(e) => onOptimizerChange(e.target.value)}
              />
              <label htmlFor="opt-roi">
                <span className="icon">📈</span>
                ROI-based
              </label>
            </div>
          </div>
          {showRangeForOptimizer && rangeData && (
            <div className="sales-range-info">
              <p className="range-label">Achievable Range ({optimizerType.toUpperCase()}):</p>
              <p className="range-values">
                {currencyFormatter.format(rangeData.min_sales)} - {currencyFormatter.format(rangeData.max_sales)}
              </p>
            </div>
          )}
        </>
      );
    }

    return (
      <div className="dialog-loading">
        <div className="loading-spinner" />
        <p>Optimizing your spend allocation...</p>
      </div>
    );
  };

  const renderActions = () => {
    if (phase === "loading") {
      return null;
    }

    return (
      <div className="dialog-actions">
        {phase === "optimizer" && (
          <button type="button" className="ghost" onClick={onBack}>
            ← Back
          </button>
        )}
        <button type="button" className="ghost" onClick={onClose}>
          Cancel
        </button>
        <button
          type="button"
          className="primary"
          onClick={phase === "target" ? onNext : onRun}
          disabled={isLoading}
        >
          {phase === "target" ? "Continue →" : "Run Optimization"}
        </button>
      </div>
    );
  };

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog-card" onClick={(e) => e.stopPropagation()}>
        <header>
          <h3>Ask eMMMy</h3>
          <button className="close-btn" onClick={onClose}>
            ✕
          </button>
        </header>
        {renderBody()}
        {error && <p className="dialog-error">{error}</p>}
        {renderActions()}
      </div>
    </div>
  );
}

AskEmmmyDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  phase: PropTypes.oneOf(["target", "optimizer", "loading"]).isRequired,
  targetSales: PropTypes.oneOfType([PropTypes.string, PropTypes.number])
    .isRequired,
  optimizerType: PropTypes.oneOf(["default", "roi"]).isRequired,
  salesRange: PropTypes.object,
  isLoadingSalesRange: PropTypes.bool,
  onTargetChange: PropTypes.func.isRequired,
  onOptimizerChange: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
  onNext: PropTypes.func.isRequired,
  onBack: PropTypes.func.isRequired,
  onRun: PropTypes.func.isRequired,
  isLoading: PropTypes.bool.isRequired,
  error: PropTypes.string,
};
