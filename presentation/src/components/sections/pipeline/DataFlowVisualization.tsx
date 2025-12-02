'use client';

import { motion } from 'framer-motion';

interface DataFlowVisualizationProps {
  isInView: boolean;
}

// Individual flowing line component
function FlowLine({
  delay,
  duration,
  startX,
  startY,
  angle,
  length,
  isInView,
}: {
  delay: number;
  duration: number;
  startX: string;
  startY: string;
  angle: number;
  length: number;
  isInView: boolean;
}) {
  return (
    <motion.div
      className="absolute"
      style={{
        left: startX,
        top: startY,
        width: length,
        height: 2,
        transform: `rotate(${angle}deg)`,
        transformOrigin: 'left center',
      }}
      initial={{ opacity: 0 }}
      animate={isInView ? { opacity: 1 } : {}}
      transition={{ duration: 0.5, delay: delay * 0.1 }}
    >
      <motion.div
        className="h-full bg-gradient-to-r from-transparent via-gray-300 to-transparent rounded-full"
        initial={{ x: '-100%' }}
        animate={isInView ? { x: '100%' } : {}}
        transition={{
          duration,
          delay: delay * 0.2,
          repeat: Infinity,
          ease: 'linear',
        }}
      />
    </motion.div>
  );
}

// Flowing dot component
function FlowDot({
  delay,
  startX,
  startY,
  endX,
  endY,
  duration,
  isInView,
}: {
  delay: number;
  startX: string;
  startY: string;
  endX: string;
  endY: string;
  duration: number;
  isInView: boolean;
}) {
  return (
    <motion.div
      className="absolute w-2 h-2 bg-gray-300 rounded-full"
      style={{ left: startX, top: startY }}
      initial={{ opacity: 0, scale: 0 }}
      animate={
        isInView
          ? {
              opacity: [0, 0.6, 0.6, 0],
              scale: [0.5, 1, 1, 0.5],
              left: [startX, endX],
              top: [startY, endY],
            }
          : {}
      }
      transition={{
        duration,
        delay,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
    />
  );
}

export default function DataFlowVisualization({ isInView }: DataFlowVisualizationProps) {
  // Flow dots configuration - converging to center
  const flowDots = [
    // From left
    { startX: '5%', startY: '30%', endX: '45%', endY: '50%', delay: 0, duration: 4 },
    { startX: '10%', startY: '50%', endX: '45%', endY: '50%', delay: 0.8, duration: 3.5 },
    { startX: '5%', startY: '70%', endX: '45%', endY: '50%', delay: 1.6, duration: 4.2 },
    // From right
    { startX: '95%', startY: '25%', endX: '55%', endY: '50%', delay: 0.4, duration: 4.1 },
    { startX: '90%', startY: '50%', endX: '55%', endY: '50%', delay: 1.2, duration: 3.8 },
    { startX: '95%', startY: '75%', endX: '55%', endY: '50%', delay: 2, duration: 4.3 },
    // From top
    { startX: '30%', startY: '5%', endX: '50%', endY: '45%', delay: 0.6, duration: 3.6 },
    { startX: '50%', startY: '5%', endX: '50%', endY: '45%', delay: 1.4, duration: 3.4 },
    { startX: '70%', startY: '5%', endX: '50%', endY: '45%', delay: 2.2, duration: 3.9 },
    // From bottom
    { startX: '35%', startY: '95%', endX: '50%', endY: '55%', delay: 1, duration: 4 },
    { startX: '65%', startY: '95%', endX: '50%', endY: '55%', delay: 1.8, duration: 3.7 },
  ];

  // Subtle radial lines from center
  const radialLines = [
    { angle: 0, length: 150, startX: '50%', startY: '50%', delay: 0, duration: 3 },
    { angle: 45, length: 120, startX: '50%', startY: '50%', delay: 1, duration: 3.5 },
    { angle: 90, length: 140, startX: '50%', startY: '50%', delay: 2, duration: 3.2 },
    { angle: 135, length: 130, startX: '50%', startY: '50%', delay: 0.5, duration: 3.8 },
    { angle: 180, length: 150, startX: '50%', startY: '50%', delay: 1.5, duration: 3.3 },
    { angle: 225, length: 125, startX: '50%', startY: '50%', delay: 2.5, duration: 3.6 },
    { angle: 270, length: 145, startX: '50%', startY: '50%', delay: 0.8, duration: 3.1 },
    { angle: 315, length: 135, startX: '50%', startY: '50%', delay: 1.8, duration: 3.4 },
  ];

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-40">
      {/* Flowing dots converging to center */}
      {flowDots.map((dot, i) => (
        <FlowDot key={`dot-${i}`} {...dot} isInView={isInView} />
      ))}

      {/* Central glow effect */}
      <motion.div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 rounded-full"
        style={{
          background: 'radial-gradient(circle, rgba(156, 163, 175, 0.3) 0%, transparent 70%)',
        }}
        initial={{ opacity: 0, scale: 0.5 }}
        animate={
          isInView
            ? {
                opacity: [0.3, 0.6, 0.3],
                scale: [0.8, 1.2, 0.8],
              }
            : {}
        }
        transition={{
          duration: 4,
          delay: 1,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />

      {/* Subtle pulsing rings */}
      {[1, 2, 3].map((ring) => (
        <motion.div
          key={`ring-${ring}`}
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 border border-gray-300 rounded-full"
          style={{
            width: ring * 80,
            height: ring * 80,
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={
            isInView
              ? {
                  opacity: [0, 0.2, 0],
                  scale: [0.8, 1.2, 0.8],
                }
              : {}
          }
          transition={{
            duration: 3 + ring * 0.5,
            delay: ring * 0.5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}
