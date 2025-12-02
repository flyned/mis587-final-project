'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';

const featureImportance = [
  { feature: 'Location (Lat/Long)', importance: 38.9 },
  { feature: 'Log Rent/SF/Yr', importance: 28.4 },
  { feature: 'Temporal Factors', importance: 15.3 },
  { feature: 'Property Density', importance: 8.9 },
  { feature: 'Missing Data Signals', importance: 4.6 },
  { feature: 'Other Features', importance: 3.9 },
];

const shapInsights = [
  {
    title: 'Location Premium',
    description: 'Properties within 2 miles of market center earn +35% rent premium',
    impact: '+35%',
    positive: true,
  },
  {
    title: 'Property Density',
    description: 'Areas with >50 properties within 5mi command +14% higher rents',
    impact: '+14%',
    positive: true,
  },
  {
    title: 'Renovation ROI',
    description: 'Properties 40+ years old: +$1.85/SF/Yr after renovation',
    impact: '+$1.85',
    positive: true,
  },
  {
    title: 'Building Age',
    description: 'Historic properties (50+ years) show higher prediction error',
    impact: '+8% MAE',
    positive: false,
  },
];

function FeatureBar({ feature, index }: { feature: typeof featureImportance[0]; index: number }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0 }}
      animate={isInView ? { opacity: 1 } : {}}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className="flex items-center gap-4"
    >
      <div className="w-40 text-sm text-gray-600">{feature.feature}</div>
      <div className="flex-1 h-6 bg-gray-100 rounded overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={isInView ? { width: `${feature.importance}%` } : {}}
          transition={{ duration: 0.8, delay: index * 0.1 + 0.2 }}
          className="h-full bg-gray-800 rounded flex items-center justify-end pr-2"
        >
          <span className="text-xs font-medium text-white">{feature.importance}%</span>
        </motion.div>
      </div>
    </motion.div>
  );
}

export default function InsightsSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section id="insights" className="section bg-gray-50">
      <div className="section-content">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <span className="text-sm text-gray-500 uppercase tracking-wide">Analysis</span>
          <h2 className="text-4xl font-bold text-black mt-2 mb-4">
            Feature Importance & SHAP
          </h2>
          <p className="text-gray-600 max-w-2xl">
            Understanding what drives property values through interpretable machine learning,
            with SHAP analysis providing transparent, actionable insights.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 mb-16">
          {/* Feature Importance Chart */}
          <div className="border border-gray-200 rounded-lg p-8 bg-white">
            <h3 className="text-lg font-semibold text-black mb-8">
              Top Feature Categories
            </h3>
            <div className="space-y-4">
              {featureImportance.map((feature, index) => (
                <FeatureBar key={feature.feature} feature={feature} index={index} />
              ))}
            </div>
            <p className="text-gray-500 text-sm mt-6">
              Location dominates (38.9%), validating the real estate mantra
              &quot;location, location, location&quot; with quantitative evidence.
            </p>
          </div>

          {/* SHAP Summary */}
          <div className="border border-gray-200 rounded-lg p-8 bg-white">
            <h3 className="text-lg font-semibold text-black mb-8">
              Key SHAP Insights
            </h3>
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-3">
                  Positive Drivers (Increase Rent)
                </h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Central location (low distance to market center)</li>
                  <li>• High property density (&gt;100 properties within 5mi)</li>
                  <li>• Premium market encoding</li>
                  <li>• Modern construction / recent FEMA dates</li>
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-3">
                  Negative Drivers (Decrease Rent)
                </h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Peripheral location (&gt;10mi from center)</li>
                  <li>• Sparse industrial areas</li>
                  <li>• Missing flood zone data</li>
                  <li>• Historic properties without renovation</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Business Insights Cards */}
        <div className="mb-16">
          <h3 className="text-xl font-semibold text-black mb-8">
            Actionable Business Insights
          </h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {shapInsights.map((insight, index) => (
              <motion.div
                key={insight.title}
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="border border-gray-200 rounded-lg p-6 bg-white"
              >
                <div className="flex items-center justify-between mb-4">
                  <span className={`text-sm font-medium ${insight.positive ? 'text-green-600' : 'text-orange-600'}`}>
                    {insight.impact}
                  </span>
                </div>
                <h4 className="text-lg font-semibold text-black mb-2">
                  {insight.title}
                </h4>
                <p className="text-gray-600 text-sm">{insight.description}</p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Example prediction explanation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="border border-gray-200 rounded-lg p-8 bg-white"
        >
          <h3 className="text-lg font-semibold text-black mb-6">
            Example: Individual Prediction Explanation
          </h3>
          <div className="max-w-2xl">
            <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200">
              <span className="text-gray-600">Base value (mean):</span>
              <span className="font-mono text-black">$12.43/SF/Yr</span>
            </div>
            <div className="space-y-3">
              {[
                { factor: 'Longitude (suburban)', impact: -0.89 },
                { factor: 'Property density', impact: 0.45 },
                { factor: 'FEMA Map Date', impact: -0.62 },
                { factor: 'Distance to center', impact: -0.34 },
                { factor: 'RBA (larger)', impact: 0.15 },
              ].map((item) => (
                <div
                  key={item.factor}
                  className="flex items-center justify-between py-2"
                >
                  <span className="text-gray-600 text-sm">{item.factor}</span>
                  <span className={`font-mono text-sm ${item.impact > 0 ? 'text-green-600' : 'text-orange-600'}`}>
                    {item.impact > 0 ? '+' : ''}${item.impact.toFixed(2)}/SF
                  </span>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-between mt-4 pt-4 border-t-2 border-black">
              <span className="text-black font-medium">Final Prediction:</span>
              <span className="font-mono text-xl text-black">$10.48/SF/Yr</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
