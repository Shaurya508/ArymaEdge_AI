import PropTypes from "prop-types";
import "./SpendSlider.css";
import { useAnimatedValue } from "../hooks/useAnimatedValue";
import WarningTooltip from "./WarningTooltip";

export default function SpendSlider({
  label,
  value,
  max,
  step = 1000,
  disabled = false,
  onChange,
  isAnimating = false,
  warnings = [],
}) {
  const [animatedValue, isValueAnimating] = useAnimatedValue(value, 1000);
  const displayValue = isValueAnimating ? animatedValue : value;

  return (
    <div className={`slider-row ${isValueAnimating || isAnimating ? 'animating' : ''}`}>
      <div className="slider-label">{label}</div>
      <div className="slider-min">0</div>
      <div className="slider-track">
        <input
          type="range"
          min={0}
          max={max}
          step={step}
          value={displayValue}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
      <div className="slider-max">{max.toLocaleString()}</div>
      <div className="slider-input">
        <input
          type="number"
          min={0}
          max={max}
          value={Math.round(displayValue)}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
      <WarningTooltip warnings={warnings}>
        <span></span>
      </WarningTooltip>
    </div>
  );
}

SpendSlider.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.number.isRequired,
  max: PropTypes.number.isRequired,
  step: PropTypes.number,
  disabled: PropTypes.bool,
  onChange: PropTypes.func.isRequired,
  isAnimating: PropTypes.bool,
  warnings: PropTypes.array,
};
