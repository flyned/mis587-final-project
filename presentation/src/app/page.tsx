'use client';

import { useEffect } from 'react';
import Navigation from '@/components/ui/Navigation';
import HeroSection from '@/components/sections/HeroSection';
import PipelineSection from '@/components/sections/PipelineSection';
import MetricsSection from '@/components/sections/MetricsSection';
import InsightsSection from '@/components/sections/InsightsSection';
import TeamSection from '@/components/sections/TeamSection';
import { ArrowUp } from 'lucide-react';

function Footer() {
  return (
    <footer className="py-12 px-6 lg:px-8 border-t border-gray-200 bg-white">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {/* Project Info */}
          <div>
            <h3 className="font-semibold text-black mb-3">
              MIS587 Final Project
            </h3>
            <p className="text-gray-600 text-sm mb-4">
              Massachusetts Industrial Properties Price Prediction using Machine Learning.
              Built with Python, TensorFlow, and deployed via Streamlit.
            </p>
            <div className="flex gap-2">
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">Python</span>
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">TensorFlow</span>
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">SHAP</span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold text-black mb-3">
              Documentation
            </h3>
            <ul className="space-y-2">
              {[
                'FINAL_REPORT.md',
                'CLAUDE.md',
                'PROJECT_STATUS.md',
              ].map((doc) => (
                <li key={doc}>
                  <span className="text-gray-600 text-sm">{doc}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Tech Stack */}
          <div>
            <h3 className="font-semibold text-black mb-3">
              Tech Stack
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {[
                'Neural Network',
                'Random Forest',
                'XGBoost',
                'LightGBM',
                'Streamlit',
                'SHAP Analysis',
              ].map((tech) => (
                <span key={tech} className="text-gray-600 text-sm">{tech}</span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-gray-200 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-gray-500 text-sm">
            Team 2 · MIS587 · Fall 2025
          </p>
          <p className="text-gray-500 text-sm">
            Presented to Lornell Real Estate · December 1, 2025
          </p>
        </div>
      </div>
    </footer>
  );
}

function ScrollToTop() {
  return (
    <button
      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      className="fixed bottom-8 right-8 z-50 w-10 h-10 rounded-full bg-black text-white flex items-center justify-center hover:bg-gray-800 transition-colors"
    >
      <ArrowUp className="w-4 h-4" />
    </button>
  );
}

export default function Home() {
  useEffect(() => {
    // Smooth scroll polyfill behavior
    document.documentElement.style.scrollBehavior = 'smooth';
  }, []);

  return (
    <main className="relative">
      <Navigation />

      <div id="hero">
        <HeroSection />
      </div>

      <PipelineSection />
      <MetricsSection />
      <InsightsSection />
      <TeamSection />

      <Footer />
      <ScrollToTop />
    </main>
  );
}
