'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import AnimatedCounter from '../animations/AnimatedCounter';
import FloatingExcelIcons from './pipeline/FloatingExcelIcons';
import DataFlowVisualization from './pipeline/DataFlowVisualization';

const stats = [
  { value: 40, suffix: '+', label: 'Excel Files' },
  { value: 15467, suffix: '', label: 'Property Records' },
  { value: 272, suffix: '', label: 'Data Columns' },
  { value: 500, suffix: '', label: 'Records/Export' },
];

export default function DataAcquisitionSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-20%' });

  return (
    <section
      ref={ref}
      className="relative min-h-screen bg-white overflow-hidden flex items-center justify-center"
    >
      {/* Background Layer: Data Flow Visualization */}
      <DataFlowVisualization isInView={isInView} />

      {/* Middle Layer: Floating Excel Icons */}
      <FloatingExcelIcons isInView={isInView} />

      {/* Content Layer */}
      <div className="relative z-10 text-center px-6 py-16 max-w-5xl mx-auto">
        {/* Step Number */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.6, ease: [0.25, 0.4, 0.25, 1] }}
          className="mb-4"
        >
          <span className="text-[10rem] md:text-[12rem] lg:text-[14rem] font-bold text-gray-100 leading-none select-none">
            01
          </span>
        </motion.div>

        {/* Title */}
        <motion.h2
          initial={{ opacity: 0, x: -50 }}
          animate={isInView ? { opacity: 1, x: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.25, 0.4, 0.25, 1] }}
          className="text-4xl md:text-5xl lg:text-6xl font-bold text-black -mt-20 md:-mt-24 lg:-mt-28 relative z-10"
        >
          Data Acquisition
        </motion.h2>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.35 }}
          className="text-xl md:text-2xl text-gray-500 mt-4 mb-16"
        >
          Collecting CoStar commercial real estate data
        </motion.p>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8 mb-16"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
              className="bg-white/80 backdrop-blur-sm border border-gray-200 rounded-xl p-6 shadow-sm"
            >
              <div className="text-4xl md:text-5xl lg:text-6xl font-bold text-black mb-2">
                {isInView ? (
                  <AnimatedCounter
                    value={stat.value}
                    suffix={stat.suffix}
                    duration={2}
                  />
                ) : (
                  '0'
                )}
              </div>
              <div className="text-sm md:text-base text-gray-500 font-medium">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Description */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 1 }}
          className="text-gray-600 text-lg max-w-2xl mx-auto leading-relaxed"
        >
          Aggregated from <span className="text-black font-semibold">40+ Excel exports</span> from
          the CoStar database, covering Massachusetts industrial properties with{' '}
          <span className="text-black font-semibold">15,467 property records</span> and{' '}
          <span className="text-black font-semibold">272 data columns</span> per property.
        </motion.p>

        {/* Data source badge */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ duration: 0.5, delay: 1.2 }}
          className="mt-8 inline-flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-full"
        >
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-sm text-gray-600">Source: CoStar Database</span>
        </motion.div>
      </div>
    </section>
  );
}
