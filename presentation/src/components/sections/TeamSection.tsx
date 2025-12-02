'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';

const team = [
  {
    name: 'Alex Siracusa',
    role: 'Lead Data Analyst',
    description: 'Feature engineering, model development, SHAP analysis',
  },
  {
    name: 'Martin Thulani Milanzi',
    role: 'Risk Analyst',
    description: 'Risk assessment, error analysis, market segmentation',
  },
  {
    name: 'Shrey Sharma',
    role: 'Project Manager',
    description: 'Project coordination, stakeholder communication, documentation',
  },
  {
    name: 'Faisal Yaseen',
    role: 'Subject Matter Expert',
    description: 'Domain expertise, business requirements, validation',
  },
];

const objectives = [
  { id: 1, title: 'Accurate Price Prediction', status: 'R² = 0.762' },
  { id: 2, title: 'Market Timing Forecast', status: '97.8% accuracy' },
  { id: 3, title: 'Value Attribution Analysis', status: 'SHAP implemented' },
  { id: 4, title: 'Opportunity Identification', status: 'Scoring system' },
];

export default function TeamSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section id="team" className="section bg-white">
      <div className="section-content">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <span className="text-sm text-gray-500 uppercase tracking-wide">Team</span>
          <h2 className="text-4xl font-bold text-black mt-2 mb-4">
            Meet Team 2
          </h2>
          <p className="text-gray-600 max-w-2xl">
            A collaborative effort combining data science expertise, business acumen,
            and domain knowledge for MIS587.
          </p>
        </motion.div>

        {/* Team members */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {team.map((member, index) => (
            <motion.div
              key={member.name}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="border border-gray-200 rounded-lg p-6"
            >
              <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                <span className="text-lg font-semibold text-gray-600">
                  {member.name.split(' ').map(n => n[0]).join('')}
                </span>
              </div>
              <h3 className="text-lg font-semibold text-black mb-1">
                {member.name}
              </h3>
              <p className="text-gray-500 text-sm mb-3">{member.role}</p>
              <p className="text-gray-600 text-sm">{member.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Sponsor and objectives */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Sponsor */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="border border-gray-200 rounded-lg p-8"
          >
            <h3 className="text-lg font-semibold text-black mb-1">
              Project Sponsor
            </h3>
            <p className="text-gray-500 text-sm mb-6">Lornell Real Estate</p>
            <div className="space-y-4">
              <div>
                <p className="text-black font-medium">Todd Lornell</p>
                <p className="text-gray-500 text-sm">Principal / Founder</p>
              </div>
              <p className="text-gray-600 text-sm">
                Commercial real estate investment and brokerage firm specializing in
                Massachusetts industrial properties. The model enables data-driven
                acquisition and valuation decisions.
              </p>
            </div>
          </motion.div>

          {/* Objectives */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="border border-gray-200 rounded-lg p-8"
          >
            <h3 className="text-lg font-semibold text-black mb-1">
              Proposal Objectives
            </h3>
            <p className="text-gray-500 text-sm mb-6">October 6, 2025</p>
            <div className="space-y-3">
              {objectives.map((obj) => (
                <div
                  key={obj.id}
                  className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0"
                >
                  <div className="flex items-center gap-3">
                    <span className="w-6 h-6 bg-black text-white text-xs rounded-full flex items-center justify-center font-medium">
                      {obj.id}
                    </span>
                    <span className="text-gray-700 text-sm">{obj.title}</span>
                  </div>
                  <span className="text-sm text-green-600 font-medium">{obj.status}</span>
                </div>
              ))}
            </div>
            <p className="text-green-600 text-sm mt-4 font-medium">
              100% Completion (4/4 Objectives)
            </p>
          </motion.div>
        </div>

        {/* Course info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-12 text-center border-t border-gray-200 pt-8"
        >
          <p className="text-gray-600 text-sm">
            MIS587 — Business Applications in Machine Learning
          </p>
          <p className="text-gray-500 text-xs mt-1">Fall 2025 · December 1, 2025</p>
        </motion.div>
      </div>
    </section>
  );
}
