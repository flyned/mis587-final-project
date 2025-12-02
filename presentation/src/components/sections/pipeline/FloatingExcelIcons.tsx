'use client';

import { motion } from 'framer-motion';
import { useMemo } from 'react';

// Excel file SVG icon component
function ExcelIcon({ size = 40 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Document shape */}
      <path
        d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z"
        fill="#E8F5E9"
        stroke="#4CAF50"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Folded corner */}
      <path
        d="M14 2V8H20"
        stroke="#4CAF50"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* X for Excel */}
      <path
        d="M9 13L15 19M15 13L9 19"
        stroke="#2E7D32"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// Generate random icon configurations
function generateIconConfigs(count: number) {
  const configs = [];
  for (let i = 0; i < count; i++) {
    configs.push({
      id: i,
      x: Math.random() * 90 + 5, // 5-95% horizontal position
      y: Math.random() * 80 + 10, // 10-90% vertical position
      size: Math.random() * 20 + 35, // 35-55px size
      opacity: Math.random() * 0.3 + 0.2, // 0.2-0.5 opacity
      floatDuration: Math.random() * 3 + 4, // 4-7s float cycle
      floatDelay: Math.random() * 2, // 0-2s initial delay
      floatAmplitude: Math.random() * 15 + 10, // 10-25px float range
      driftX: Math.random() * 20 - 10, // -10 to 10px horizontal drift
      entranceDelay: i * 0.1, // Staggered entrance
    });
  }
  return configs;
}

interface FloatingExcelIconsProps {
  isInView: boolean;
}

export default function FloatingExcelIcons({ isInView }: FloatingExcelIconsProps) {
  // Memoize icon configurations to prevent regeneration on re-render
  const iconConfigs = useMemo(() => generateIconConfigs(14), []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {iconConfigs.map((config) => (
        <motion.div
          key={config.id}
          className="absolute"
          style={{
            left: `${config.x}%`,
            top: `${config.y}%`,
            opacity: config.opacity,
          }}
          initial={{ opacity: 0, scale: 0.3, y: 30 }}
          animate={
            isInView
              ? {
                  opacity: config.opacity,
                  scale: 1,
                  y: [0, -config.floatAmplitude, 0],
                  x: [0, config.driftX, 0],
                }
              : {}
          }
          transition={{
            opacity: { duration: 0.6, delay: config.entranceDelay + 0.4 },
            scale: { duration: 0.6, delay: config.entranceDelay + 0.4 },
            y: {
              duration: config.floatDuration,
              delay: config.floatDelay + config.entranceDelay,
              repeat: Infinity,
              repeatType: 'reverse',
              ease: 'easeInOut',
            },
            x: {
              duration: config.floatDuration * 1.5,
              delay: config.floatDelay + config.entranceDelay,
              repeat: Infinity,
              repeatType: 'reverse',
              ease: 'easeInOut',
            },
          }}
        >
          <ExcelIcon size={config.size} />
        </motion.div>
      ))}
    </div>
  );
}
