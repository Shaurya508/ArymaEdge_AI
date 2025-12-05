import PropTypes from "prop-types";
import { useState, useRef, useEffect } from "react";
import "./WarningTooltip.css";

export default function WarningTooltip({ warnings, children }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const indicatorRef = useRef(null);
  const tooltipRef = useRef(null);
  const hideTimeoutRef = useRef(null);

  useEffect(() => {
    if (showTooltip && indicatorRef.current) {
      const rect = indicatorRef.current.getBoundingClientRect();
      setTooltipPosition({
        top: rect.top - 10,
        left: rect.left + rect.width / 2,
      });
    }
  }, [showTooltip]);

  const handleMouseEnter = () => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
    }
    setShowTooltip(true);
  };

  const handleMouseLeave = () => {
    // Delay hiding to allow mouse to move to tooltip
    hideTimeoutRef.current = setTimeout(() => {
      setShowTooltip(false);
    }, 100);
  };

  useEffect(() => {
    return () => {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
    };
  }, []);

  if (!warnings || warnings.length === 0) {
    return <>{children}</>;
  }

  // Group warnings by type
  const errorWarnings = warnings.filter((w) => w.type === "error");
  const hasError = errorWarnings.length > 0;

  return (
    <div className="warning-tooltip-wrapper">
      <div 
        ref={indicatorRef}
        className={`warning-indicator ${hasError ? 'error' : 'warning'}`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        !
      </div>
      {children}
      {showTooltip && (
        <div 
          ref={tooltipRef}
          className="warning-tooltip visible"
          style={{
            top: `${tooltipPosition.top}px`,
            left: `${tooltipPosition.left}px`,
            transform: 'translate(-50%, -100%)',
          }}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          <div className="tooltip-content">
            {warnings.map((warning, index) => (
              <div key={index} className={`tooltip-warning-item ${warning.type}`}>
                <div className="tooltip-warning-message">{warning.message}</div>
                {warning.detail && (
                  <div className="tooltip-warning-detail">{warning.detail}</div>
                )}
              </div>
            ))}
          </div>
          <div className="tooltip-arrow"></div>
        </div>
      )}
    </div>
  );
}

WarningTooltip.propTypes = {
  warnings: PropTypes.array,
  children: PropTypes.node.isRequired,
};

