'use client';

import { motion } from 'framer-motion';
import dynamic from 'next/dynamic';
import { ChevronDown } from 'lucide-react';

const NeuralNetwork = dynamic(() => import('../3d/NeuralNetwork'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="w-12 h-12 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
    </div>
  ),
});

export default function HeroSection() {
  return (
    <section className="relative min-h-screen bg-white">
      {/* Two-column layout: Text left, 3D right */}
      <div className="min-h-screen flex flex-col lg:flex-row">
        {/* Left side - Text content */}
        <div className="flex-1 flex items-center px-8 lg:px-16 py-24 lg:py-0">
          <div className="max-w-xl">
            {/* Course badge */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="mb-8"
            >
              <span className="text-sm text-gray-500 tracking-wide uppercase">
                MIS587 — Business Applications in Machine Learning
              </span>
            </motion.div>

            {/* Main title - clear hierarchy, no overlap */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="mb-6"
            >
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-black leading-tight">
                Massachusetts Industrial Properties
              </h1>
            </motion.div>

            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.15 }}
              className="text-2xl md:text-3xl font-medium text-gray-600 mb-8"
            >
              Price Prediction Model
            </motion.h2>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-lg text-gray-600 mb-12"
            >
              A machine learning approach to commercial real estate valuation,
              achieving <span className="text-black font-semibold">76.2% accuracy</span> with
              neural networks on 12,534 properties.
            </motion.p>

            {/* Key metrics - simple row */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12"
            >
              {[
                { label: 'Properties', value: '12,534' },
                { label: 'Features', value: '194' },
                { label: 'R² Score', value: '0.762' },
                { label: 'DOM Accuracy', value: '97.8%' },
              ].map((item, index) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.4, delay: 0.4 + index * 0.1 }}
                >
                  <div className="text-3xl font-bold text-black">{item.value}</div>
                  <div className="text-sm text-gray-500 mt-1">{item.label}</div>
                </motion.div>
              ))}
            </motion.div>

            {/* Team info */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="border-t border-gray-200 pt-6"
            >
              <p className="text-gray-700">Team 2 · December 2025</p>
              <p className="text-gray-500 text-sm mt-1">Presented to Lornell Real Estate</p>
            </motion.div>
          </div>
        </div>

        {/* Right side - 3D Neural Network (contained, no overlap) */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="flex-1 relative min-h-[400px] lg:min-h-0 bg-gray-50"
        >
          <div className="absolute inset-0">
            <NeuralNetwork />
          </div>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="absolute bottom-8 left-8 lg:left-16"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="flex items-center gap-2 text-gray-400"
        >
          <ChevronDown className="w-5 h-5" />
          <span className="text-sm">Scroll</span>
        </motion.div>
      </motion.div>
    </section>
  );
}
