import { useEffect, useState, useRef } from 'react';

export function useAnimatedValue(targetValue, duration = 800) {
  const [currentValue, setCurrentValue] = useState(targetValue);
  const [isAnimating, setIsAnimating] = useState(false);
  const rafRef = useRef(null);
  const startTimeRef = useRef(null);
  const startValueRef = useRef(targetValue);

  useEffect(() => {
    // If target changed, start animation
    if (targetValue !== currentValue) {
      setIsAnimating(true);
      startValueRef.current = currentValue;
      startTimeRef.current = null;

      const animate = (timestamp) => {
        if (!startTimeRef.current) {
          startTimeRef.current = timestamp;
        }

        const elapsed = timestamp - startTimeRef.current;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out cubic)
        const easeProgress = 1 - Math.pow(1 - progress, 3);

        const newValue = startValueRef.current + 
          (targetValue - startValueRef.current) * easeProgress;

        setCurrentValue(newValue);

        if (progress < 1) {
          rafRef.current = requestAnimationFrame(animate);
        } else {
          setCurrentValue(targetValue);
          setIsAnimating(false);
        }
      };

      rafRef.current = requestAnimationFrame(animate);

      return () => {
        if (rafRef.current) {
          cancelAnimationFrame(rafRef.current);
        }
      };
    }
  }, [targetValue, duration]);

  return [currentValue, isAnimating];
}

