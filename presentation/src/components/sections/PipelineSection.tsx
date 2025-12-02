'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { ArrowRight } from 'lucide-react';
import DataAcquisitionSection from './DataAcquisitionSection';

// Pipeline steps 02-06 (Step 01 is now a full-page cinematic section)
const pipelineSteps = [
  {
    number: '02',
    title: 'Data Cleaning',
    subtitle: '68% Column Reduction',
    description: 'Deduplication, null handling, and quality filtering',
    metrics: ['272 → 87 features', '2,469 duplicates removed'],
  },
  {
    number: '03',
    title: 'Train/Val/Test Split',
    subtitle: '60/20/20 Stratified',
    description: 'Zero data leakage with geographic stratification',
    metrics: ['7,520 train', '2,507 val/test each'],
  },
  {
    number: '04',
    title: 'Feature Engineering',
    subtitle: '6-Phase Pipeline',
    description: 'Temporal, geospatial, numerical, categorical transformations',
    metrics: ['78 → 194 features', '5-fold CV encoding'],
  },
  {
    number: '05',
    title: 'Model Training',
    subtitle: '7 Algorithms',
    description: 'Neural Network, Random Forest, XGBoost, LightGBM + baselines',
    metrics: ['R² = 0.762', 'MAE = $0.99/SF'],
  },
  {
    number: '06',
    title: 'Deployment',
    subtitle: '7-Page Dashboard',
    description: 'Interactive Streamlit app with SHAP explanations',
    metrics: ['Real-time predictions', 'Batch processing'],
  },
];

function PipelineStep({
  step,
  index,
}: {
  step: typeof pipelineSteps[0];
  index: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className="border-b border-gray-200 py-8 last:border-b-0"
    >
      <div className="flex gap-6">
        {/* Step number */}
        <div className="flex-shrink-0">
          <span className="text-4xl font-bold text-gray-300">{step.number}</span>
        </div>

        {/* Content */}
        <div className="flex-1">
          <div className="flex items-baseline gap-3 mb-2">
            <h3 className="text-xl font-semibold text-black">
              {step.title}
            </h3>
            <span className="text-sm text-gray-500">
              {step.subtitle}
            </span>
          </div>
          <p className="text-gray-600 mb-4">{step.description}</p>

          {/* Metrics */}
          <div className="flex flex-wrap gap-3">
            {step.metrics.map((metric, i) => (
              <span
                key={i}
                className="text-sm text-gray-700 bg-gray-100 px-3 py-1 rounded"
              >
                {metric}
              </span>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default function PipelineSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section id="pipeline">
      {/* Step 01: Full-page cinematic Data Acquisition section */}
      <DataAcquisitionSection />

      {/* Steps 02-06: Pipeline continuation */}
      <div className="section bg-white">
        <div className="section-content">
          {/* Header */}
          <motion.div
            ref={ref}
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
            className="mb-12"
          >
            <span className="text-sm text-gray-500 uppercase tracking-wide">Methodology</span>
            <h2 className="text-4xl font-bold text-black mt-2 mb-4">
              End-to-End ML Pipeline
            </h2>
            <p className="text-gray-600 max-w-2xl">
              From raw CoStar exports to production-ready predictions through
              five more carefully engineered stages.
            </p>
          </motion.div>

          {/* Pipeline steps 02-06 */}
          <div className="max-w-3xl">
            {pipelineSteps.map((step, index) => (
              <PipelineStep
                key={step.title}
                step={step}
                index={index}
              />
            ))}
          </div>

          {/* Data flow summary */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-16 border border-gray-200 rounded-lg p-8 max-w-4xl"
          >
            <h3 className="text-lg font-semibold text-black mb-6">
              Data Transformation Summary
            </h3>
            <div className="flex flex-wrap items-center gap-6">
              {[
                { label: 'Raw Data', value: '15,467 rows' },
                { label: 'Cleaned', value: '12,534 unique' },
                { label: 'Features', value: '194 engineered' },
                { label: 'Model', value: 'Neural Network' },
                { label: 'Result', value: 'R² = 0.762' },
              ].map((item, index, arr) => (
                <div key={item.label} className="flex items-center gap-6">
                  <div>
                    <div className="text-2xl font-bold text-black">{item.value}</div>
                    <div className="text-sm text-gray-500">{item.label}</div>
                  </div>
                  {index < arr.length - 1 && (
                    <ArrowRight className="w-5 h-5 text-gray-400" />
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
