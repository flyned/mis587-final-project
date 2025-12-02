'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import AnimatedCounter from '../animations/AnimatedCounter';

const models = [
  {
    name: 'Neural Network',
    valR2: 0.762,
    testR2: 0.615,
    mae: 0.99,
    within10: 72.4,
    isChampion: true,
  },
  {
    name: 'Random Forest',
    valR2: 0.676,
    testR2: 0.656,
    mae: 1.20,
    within10: 65.3,
    isChampion: false,
  },
  {
    name: 'LightGBM',
    valR2: 0.671,
    testR2: 0.651,
    mae: 1.22,
    within10: 64.8,
    isChampion: false,
  },
];

function ModelCard({ model, index }: { model: typeof models[0]; index: number }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className={`border rounded-lg p-6 ${
        model.isChampion ? 'border-black bg-gray-50' : 'border-gray-200'
      }`}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-black">{model.name}</h3>
        {model.isChampion && (
          <span className="text-xs font-medium text-white bg-black px-2 py-1 rounded">
            Best Model
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-2xl font-bold text-black">
            <AnimatedCounter value={model.valR2} decimals={3} />
          </div>
          <div className="text-sm text-gray-500">Validation R²</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-gray-600">
            <AnimatedCounter value={model.testR2} decimals={3} />
          </div>
          <div className="text-sm text-gray-500">Test R²</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-black">
            $<AnimatedCounter value={model.mae} decimals={2} />
          </div>
          <div className="text-sm text-gray-500">MAE ($/SF/Yr)</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-black">
            <AnimatedCounter value={model.within10} decimals={1} suffix="%" />
          </div>
          <div className="text-sm text-gray-500">Within ±10%</div>
        </div>
      </div>
    </motion.div>
  );
}

export default function MetricsSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section id="metrics" className="section bg-white">
      <div className="section-content">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <span className="text-sm text-gray-500 uppercase tracking-wide">Results</span>
          <h2 className="text-4xl font-bold text-black mt-2 mb-4">
            Model Performance
          </h2>
          <p className="text-gray-600 max-w-2xl">
            Comprehensive evaluation across multiple algorithms, with Neural Network
            emerging as the best model for Massachusetts industrial properties.
          </p>
        </motion.div>

        {/* Key highlights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-16 border-y border-gray-200 py-8"
        >
          <div>
            <div className="text-4xl font-bold text-black">76.2%</div>
            <div className="text-sm text-gray-500 mt-1">Validation Accuracy</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-black">$0.99</div>
            <div className="text-sm text-gray-500 mt-1">MAE ($/SF/Yr)</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-black">97.8%</div>
            <div className="text-sm text-gray-500 mt-1">DOM Accuracy</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-black">72.4%</div>
            <div className="text-sm text-gray-500 mt-1">Within ±10%</div>
          </div>
        </motion.div>

        {/* Model comparison */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          {models.map((model, index) => (
            <ModelCard key={model.name} model={model} index={index} />
          ))}
        </div>

        {/* Business interpretation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="border border-gray-200 rounded-lg p-8"
        >
          <h3 className="text-lg font-semibold text-black mb-6">
            Business Interpretation
          </h3>
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <div className="text-3xl font-bold text-black">$12.43</div>
              <div className="text-gray-600 mt-1">Mean Rent/SF/Yr</div>
              <div className="text-sm text-gray-500 mt-1">Dataset average</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-black">±$0.99</div>
              <div className="text-gray-600 mt-1">Expected Error</div>
              <div className="text-sm text-gray-500 mt-1">Neural Network MAE</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-black">±8%</div>
              <div className="text-gray-600 mt-1">Percentage Error</div>
              <div className="text-sm text-gray-500 mt-1">Beats traditional appraisals</div>
            </div>
          </div>
          <p className="text-gray-500 text-sm mt-6 pt-6 border-t border-gray-200">
            For a 100,000 SF property, expected annual rent prediction error is ±$99,000.
            This exceeds traditional appraisal accuracy of 10-20%.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
