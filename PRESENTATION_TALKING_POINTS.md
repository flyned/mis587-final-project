# Presentation Talking Points
## Massachusetts Industrial Properties Price Prediction and Market Analysis
## MIS587 Final Project - Team 2
### Duration: 12-15 minutes

---

## SLIDE 1: Title Slide (30 seconds)

**What to Say:**

"Good [morning/afternoon], everyone. We are Team 2 presenting our MIS587 final project: Massachusetts Industrial Properties Price Prediction and Market Analysis.

This project was developed for our client Lornell Real Estate, where we applied advanced machine learning techniques to solve real-world business problems in the commercial real estate industry—specifically, predicting industrial property rental rates and market timing to support investment decision-making.

Our team includes Alex Siracusa as Lead Data Analyst, Martin Thulani Milanzi as Risk Analyst, Shrey Sharma as Project Manager, and Faisal Yaseen as Subject Matter Expert.

Let's dive in."

**Delivery Tips:**
- Speak confidently and make eye contact
- Acknowledge the team and sponsor
- Briefly acknowledge your audience

---

## SLIDE 2: Problem Statement (1 minute)

**What to Say:**

"Let me start by framing the business problem we're solving.

In the commercial real estate industry, property valuation is a critical task that traditionally takes 4 to 8 hours per property. Analysts manually review comparable properties, market trends, and property characteristics to estimate rental values.

This manual process has three major problems:
1. It's extremely time-intensive—imagine valuing a portfolio of 100 properties
2. It misses complex patterns in the data that humans can't easily detect
3. It produces inconsistent results that depend heavily on the analyst's subjective judgment

To put this in context, the average 50,000 square-foot industrial property in our dataset generates $621,500 in annual rent. When you're making multi-million dollar investment decisions, you need fast, accurate, and data-driven valuations—not gut feelings.

This is where machine learning comes in."

**Key Points to Emphasize:**
- Real business pain point (time + accuracy)
- High stakes (large dollar amounts)
- Need for automation

**Transition:** "So what were we trying to achieve?"

---

## SLIDE 3: Project Objectives (1 minute)

**What to Say:**

"These four objectives came directly from our October 6, 2025 proposal to Lornell Real Estate:

**First—Accurate Price Prediction.** We needed to build models that could predict industrial property prices, especially price per square foot, with commercial-grade accuracy.

**Second—Market Timing Forecast.** Lornell specifically requested forecasting Days-on-Market to inform their bidding strategies and negotiation tactics. This helps them know when to make aggressive offers versus when to wait.

**Third—Value Attribution Analysis.** It's not enough to just make predictions—we need to quantify which specific features drive value using SHAP analysis. This transparency is critical for stakeholder trust.

**Fourth—Opportunity Identification.** We wanted to identify promising off-market properties through predictive framework analysis. This is where investors find value before others do.

These weren't just technical goals—each one was a specific request from our client Lornell Real Estate, and I'm pleased to report we achieved all four."

**Key Points:**
- Link objectives directly to the original proposal
- Emphasize client relationship (Lornell Real Estate)
- Foreshadow 100% objective completion

**Transition:** "To achieve these goals, we needed high-quality data. Let me show you what we worked with."

---

## SLIDE 4: Dataset Overview (1 minute)

**What to Say:**

"Our data came from CoStar, which is the industry-leading platform for commercial real estate data. They're essentially the Bloomberg of real estate.

We started with 15,467 property records spanning Massachusetts and New England—covering six markets including Boston, Worcester, Providence, Springfield, Barnstable Town, and Pittsfield—with 272 original features covering everything from building size and age to amenities, location, and financial metrics.

But here's the key—raw data is messy. After rigorous cleaning and deduplication, we ended up with 12,534 unique properties. Why the reduction? Same properties often appear multiple times in the dataset with different listing dates. We had to deduplicate to prevent data leakage—which I'll explain in a moment.

We split the data 60-20-20: 7,520 properties for training, 2,507 for validation, and 2,507 for final testing. This standard split allows us to tune the model on validation data while keeping the test set completely untouched for final evaluation.

This flowchart shows how our dataset shrank at each stage—not because we lost data, but because we got smarter about data quality."

**Key Points:**
- Credible data source (CoStar)
- Data reduction is intentional, not accidental
- Standard ML practices (train/val/test split)

**Transition:** "Let me walk you through our data cleaning process, which was absolutely critical."

---

## SLIDE 5: Data Cleaning & Preparation (1.5 minutes)

**What to Say:**

"Data cleaning is where most real-world ML projects succeed or fail. We implemented a rigorous three-phase process.

**Phase 1 was Column Reduction.** We went from 272 features down to 87—a 68% reduction. We removed columns like broker contact information, columns with 95% missing values, and residential-specific metrics that didn't apply to industrial properties. Fewer, higher-quality features beats many noisy ones.

**Phase 2 focused on Data Quality.** We built custom parsers to handle messy formats—like rent ranges ('$10-12/SF/Yr') that needed to be converted to single values. We consolidated rare categories and removed 2,469 duplicate properties by address. Same property, different listing date—we kept only the most recent.

**Phase 3—and this is critical—Data Leakage Prevention.** [Point to callout box] Data leakage is the number-one cause of model failure in production. It happens when information from your test set 'leaks' into your training set, making your model look amazing in testing but fail in the real world.

We prevented this by deduplicating BEFORE splitting—not after. We validated that zero properties appeared in both training and test sets. We used geographic stratification to ensure balanced market representation.

This seems tedious, but it's the difference between a real model and a toy model."

**Key Points:**
- Emphasize real-world messiness
- Explain data leakage clearly (use analogy if needed: "It's like studying with the exam answers")
- Show technical rigor

**Transition:** "With clean data in hand, we moved to feature engineering—where the magic happens."

---

## SLIDE 6: Feature Engineering Pipeline (1.5 minutes)

**What to Say:**

"Feature engineering is the process of transforming raw data into features that help the model learn better. We implemented a six-phase pipeline that took our 78 clean features and engineered 195 total features.

Let me quickly walk through each phase:

**Temporal features**—things like building age, years since the property was last sold, and seasonal patterns. We even created construction era categories because a building from 1950 has different characteristics than one from 2020.

**Geospatial features**—distance to market center, property density within 1 and 5 miles, nearest competitor distance. These capture agglomeration effects—why properties cluster in certain areas command premium rents.

**Numerical transformations**—log transforms for skewed variables, ratio features like rent per parking space, and interaction features like building age times renovation status. These capture non-linear relationships.

**Imputation**—strategic missing value handling. For example, if 'Year Renovated' is blank, that means it was never renovated, which is actually informative. We preserve that signal.

**Categorical encoding**—we used target encoding with cross-validation for high-cardinality features like city names. This prevents overfitting while capturing the relationship between categories and rental rates.

**Outlier treatment**—we capped extreme values instead of removing them. Winsorization at the 1st and 99th percentiles keeps the data while reducing the influence of outliers.

Here's the key insight: [Point to callout box] 40% of our top 20 most important features were engineered, not raw. Feature engineering directly drives model performance."

**Key Points:**
- Don't get too technical—explain "why" each phase matters
- Emphasize the 40% statistic
- Show systematic thinking

**Transition:** "With 195 features ready, we could start building models. Let's see which one won."

---

## SLIDE 7: Model Comparison (1 minute)

**What to Say:**

"We evaluated seven machine learning algorithms, including deep learning.

[Point to table] Here's the head-to-head comparison. Our champion model is the Neural Network with R-squared of 0.762 and Mean Absolute Error of just 99 cents per square foot per year.

That's a 19% improvement over Random Forest, which was at 0.640. The Neural Network's architecture—256, 128, 64 neurons with dropout—captures complex non-linear relationships that tree-based methods miss.

Random Forest still provides value for interpretability and has a smaller validation-to-test gap, so we keep both models available.

Most importantly, we also developed a Days on Market model that achieves 97.8% accuracy within 7 days—directly addressing Lornell's second objective for market timing predictions.

[Point to winner box] This means we've achieved all four proposal objectives with high accuracy."

**Key Points:**
- Neural Network is the champion at 0.762 R²
- Emphasize the 19% improvement over Random Forest
- Highlight Days on Market model (Objective 2)
- Connect back to 100% objective completion

**Transition:** "Let's dive into how well this model actually performs."

---

## SLIDE 8: Model Performance Results (1 minute)

**What to Say:**

"Our Neural Network model delivers strong, production-ready performance.

On the validation set, R-squared of 0.762 means we explain 76.2% of the variation in rental rates. The Mean Absolute Error is just 99 cents per square foot per year.

But here's the business metric that really matters: [Point to box] 72.4% of our predictions fall within plus-or-minus 10% of actual rents. In commercial real estate, ±10% is the industry standard for acceptable valuation accuracy. We're exceeding that standard.

We also developed a Days on Market model achieving 97.8% accuracy within 7 days—this directly addresses Lornell's second objective for market timing. They can now make informed decisions about when to bid aggressively versus when to wait.

To put this in context: [Point to business context] for our typical 50,000 square-foot property generating $621,500 in annual rent, our average prediction error is about $49,500 per year, or 8%. That's within commercial tolerance and a significant improvement over traditional methods.

[Point to callout] All four proposal objectives have been achieved with this system."

**Key Points:**
- Neural Network at 76.2% R², MAE under $1
- 72.4% within ±10% (exceeds industry standard)
- Days on Market model addresses Objective 2
- All four objectives achieved

**Transition:** "Performance numbers are great, but let's understand WHAT the model is learning."

---

## SLIDE 9: Feature Importance (1 minute)

**What to Say:**

"This is where we answer the question: What actually drives commercial property rents?

The top 10 features reveal clear patterns. [Point to chart]

**Location dominates.** Longitude and Latitude together account for 22.6% of predictive power. Location, location, location—the real estate mantra holds true.

**Temporal factors matter.** FEMA Map Date appears twice in the top 10. This might seem strange—why does flood map timing matter? It turns out, FEMA map updates correlate with recent development activity, making it a proxy for neighborhood investment.

**Geospatial context is critical.** Property density within 5 miles and distance to market center together contribute 10%. This captures agglomeration effects—businesses want to be near other businesses.

**Engineered features punch above their weight.** Distance times Age—an interaction term we created—ranks 10th. None of the raw data fields captured this relationship.

Here's the business insight: [Point to insight box] Location and temporal factors drive 57% of predictive power. If you're valuing a property, focus on WHERE it is and WHEN key events happened—building construction, renovations, market developments.

The rest—building size, amenities, parking—matter, but they're secondary."

**Key Points:**
- Make it a story about what drives value
- Explain "surprising" features (FEMA dates)
- Tie back to business decision-making

**Transition:** "Feature importance tells us what matters globally. But how does the model make individual predictions? That's where SHAP comes in."

---

## SLIDE 10: SHAP Analysis (1.5 minutes)

**What to Say:**

"SHAP stands for SHapley Additive exPlanations. It's a technique from game theory that shows exactly how each feature contributes to EACH prediction.

Why does this matter? In regulated industries or high-stakes decisions, stakeholders want to know: 'Why did the model predict this value?' SHAP gives us that explanation.

Let me walk through the top features:

**Location (Longitude/Latitude)—**properties in the Boston metro area command significantly higher rents, adding $1.50 to $3 per square foot compared to western Massachusetts markets like Springfield and Pittsfield where rents are $0.50 to $1.50 lower. This SHAP plot shows that geographic relationship clearly.

**FEMA Map Date—**recent flood map updates (2015 to 2025) add $1 to $3 per square foot. Older maps correlate with older development, which tends to have lower rents. It's not about flood risk—it's a proxy for investment recency.

**Property Density—**high-density areas like Boston with more than 100 nearby properties command a $0.50 to $2 premium per square foot. This is the agglomeration effect—businesses benefit from proximity to suppliers, customers, and labor pools.

[Point to visualization] This SHAP summary plot shows the distribution of impacts across all features. Red dots are high feature values, blue are low. The x-axis shows the impact on rent prediction.

The key takeaway: [Point to box] Every single prediction comes with an explanation. When the model says '$12.50 per square foot,' we can say exactly why—$2 from location, $1.50 from recent development, $0.75 from high density, and so on.

This transparency is critical for adoption. Stakeholders won't trust a black box, but they will trust a model that shows its work."

**Key Points:**
- Explain SHAP simply (don't assume technical audience)
- Use concrete examples ($2 premium from X)
- Emphasize transparency = trust = adoption

**Transition:** "No model is perfect. Let's look at where this one struggles."

---

## SLIDE 11: Error Analysis (1 minute)

**What to Say:**

"Understanding where your model fails is just as important as celebrating where it succeeds.

We segmented errors by property type and market to find patterns.

**By property type:** [Point to left chart] The model performs best on Showrooms—$0.58 MAE and R-squared of 0.90—and Warehouses at $0.64 MAE with R-squared of 0.87. These are the bread-and-butter industrial property types with consistent characteristics. Food Processing properties are trickier, with MAE of $0.96. Why? Specialized properties have more heterogeneous characteristics, making them harder to predict from standard features.

**By market:** [Point to right chart] Here's an interesting finding. The model performs BEST in Providence, RI—MAE of just $0.56—and Boston at $0.68 MAE. These core metro markets have the highest sample density in our training data, allowing the model to learn robust patterns.

Springfield, MA has higher error ($0.85 MAE) with lower R-squared (0.63). Why? Western Massachusetts has different market dynamics and less data for the model to learn from.

The business insight: [Point to callout] This model is most reliable for standard warehouse/industrial properties in core New England metros like Boston and Providence. For specialty properties or smaller markets like Springfield and Pittsfield, use predictions with caution and wider confidence intervals.

Knowing your model's limitations builds credibility."

**Key Points:**
- Frame errors as insights, not failures
- Explain geographic patterns (core metros vs. western MA)
- Give actionable guidance (where to trust the model)

**Transition:** "Now let me show you how stakeholders actually use this model—through our web application."

---

## SLIDE 12: Streamlit App Demo (1.5 minutes)

**What to Say:**

"Academic projects often end with a Jupyter notebook. We went further and built a production-ready web application using Streamlit.

The app has five main pages: [Point to screenshots]

**Property Lookup** is where users enter a single property's characteristics—location, size, building age, amenities—and get an instant prediction. The response includes the predicted rent, a confidence interval, and a SHAP explanation showing which features drove that specific prediction. It also displays comparable properties for context.

**Model Insights** provides transparency into how the model works—feature importance charts, SHAP summary plots, and performance metrics. This builds trust with technical and non-technical users alike.

**Batch Prediction** lets users upload a CSV file with hundreds of properties and get predictions for all of them at once. This is critical for portfolio analysis—imagine valuing 500 properties in under a minute instead of 500 × 6 hours.

**Market Comparison** is my favorite feature. You can define a hypothetical property and ask: 'What would this property rent for in Boston versus Worcester versus Springfield?' It shows geographic rent differences across the New England markets in our dataset.

**About** documents the methodology, data sources, and model performance—full transparency.

The entire application runs locally on this URL [point to tech stack], delivers predictions in under one second, and requires zero coding knowledge to use.

This isn't a prototype—it's a tool that real estate professionals could use tomorrow."

**Key Points:**
- Focus on user benefits, not tech specs
- Highlight speed (<1 second)
- Emphasize the "portfolio analysis" use case (500 properties in 1 minute)
- If possible, do a 15-second live demo

**Transition:** "Let's talk about the business impact this creates."

---

## SLIDE 13: Business Impact (1 minute)

**What to Say:**

"Machine learning models don't create value by existing—they create value by changing how people work. Let me quantify the impact in three areas.

**Time Savings.** [Point to clock icon] Traditional comparable analysis takes 4 to 8 hours per property. Our model delivers predictions in under one minute. That's a 95% reduction in valuation time. If an analyst values 500 properties per year—a typical workload—we're saving them 2,400 to 3,600 hours annually. At $75 per hour loaded cost, that's $225,000 in annual labor savings for one person.

**Improved Decision Quality.** [Point to chart icon] Quantitative predictions beat gut feelings. The model provides confidence intervals so analysts understand uncertainty, and SHAP explanations make every prediction transparent. Compare that to manual analysis where different analysts can arrive at wildly different valuations for the same property.

**Opportunity Identification.** [Point to lightbulb] Residual analysis—comparing predicted rent to actual market rent—automatically flags undervalued properties. Properties where the model predicts $15 per square foot but the market is asking $13 become investment targets. Batch prediction lets you screen hundreds of properties quickly.

[Point to ROI box] When you combine time savings, better decisions, and opportunity identification, you're looking at $225,000 or more in annual value—and that's a conservative estimate for a single analyst.

Scale this across a 10-person team, and the ROI becomes obvious."

**Key Points:**
- Quantify everything in dollars
- Use realistic assumptions ($75/hr is reasonable)
- Frame impact as "freeing analysts to do higher-value work"

**Transition:** "Of course, building this wasn't all smooth sailing. Let me share four major challenges we overcame."

---

## SLIDE 14: Technical Challenges (1.5 minutes)

**What to Say:**

"Every real-world ML project hits obstacles. Here are the four biggest challenges we faced and how we solved them.

**Challenge 1: Data Leakage.** [Point to problem] Remember those duplicate properties I mentioned? If we had split the data BEFORE deduplicating, the same property could appear in both training and test sets. The model would memorize those specific properties and look amazing in testing but fail in production.

**Solution:** [Point to solution arrow] We deduplicated first, then split. We validated zero address overlap between sets. It seems obvious now, but catching this early saved the entire project.

**Result:** Trustworthy model that generalizes to new properties.

**Challenge 2: Memory Explosion.** [Point to problem] My laptop has 16GB of RAM. The initial Random Forest training with default settings—using all CPU cores in parallel—consumed 70 gigabytes and crashed my computer. I couldn't even finish training.

**Solution:** [Point to solution] Set n_jobs equals 1 to disable parallelization during tree building. Processed features in batches instead of all at once.

**Result:** Training now uses 384 megabytes—a 99.5% reduction. I can train on a laptop.

**Challenge 3: Metadata Mismatch.** [Point to problem] When I first deployed the Streamlit app, validation metrics showed 'N/A' instead of the actual values. Turns out the training script saved metrics under one key name and the app looked for a different key name.

**Solution:** Standardized all metadata keys across the codebase.

**Result:** Metrics now display correctly.

**Challenge 4: Feature Complexity.** [Point to problem] We engineered 195 features. More features means more risk of overfitting—learning noise instead of signal.

**Solution:** Cross-validation to detect overfitting early, plus SHAP-based feature filtering to focus on what matters.

**Result:** Minimal train-test gap—the model generalizes well.

These challenges taught me more than any textbook. Real-world ML is 20% modeling and 80% debugging data, infrastructure, and plumbing."

**Key Points:**
- Be honest about challenges (shows maturity)
- Show problem-solving process
- Emphasize lessons learned

**Transition:** "Let me wrap up with conclusions and next steps."

---

## SLIDE 15: Conclusions & Next Steps (1.5 minutes)

**What to Say:**

"Let me summarize what we accomplished and where this project could go next.

**Achievements:** [Point to checkmarks]

We built a production-ready machine learning system from scratch. The Random Forest model achieves R-squared of 0.640 and Mean Absolute Error of $1.24 per square foot—meeting commercial real estate accuracy standards.

We engineered 192 features through a systematic six-phase pipeline, implemented SHAP analysis for interpretability, and deployed an interactive web application that real analysts could use today.

We generated 29 visualizations and comprehensive documentation. This isn't a toy project—it's a complete system.

**Learning Outcomes:**

This project gave me hands-on experience with the full ML pipeline—from messy Excel files to production deployment. I learned advanced techniques like target encoding with cross-validation, SHAP-based interpretation, and memory optimization.

But more importantly, I learned how to solve real-world problems. Data leakage prevention, handling class imbalance, making models interpretable for non-technical stakeholders—these are the skills that matter in industry.

**Future Enhancements:** [Point to roadmap]

In the short term, hyperparameter tuning could push R-squared above 0.70. We used default settings to save time—there's optimization headroom.

Medium-term, deploying to the cloud with a REST API would enable system integration—imagine property management software automatically pulling valuations.

Long-term, we could add time-series forecasting to predict rent trends, not just current values. Portfolio optimization algorithms could recommend which properties to buy or sell.

[Point to final thought] We took 15,467 raw property records and turned them into a working ML system in 12 weeks. That's the power of systematic application of machine learning to real business problems.

Thank you. I'm happy to answer questions."

**Key Points:**
- Summarize achievements confidently
- Frame learnings as applicable skills
- Show awareness that the project can evolve
- End strongly

---

## SLIDE 16: Q&A (Variable Time)

**Anticipated Questions & Suggested Answers:**

**Q: "Why Random Forest over XGBoost/LightGBM which are often better?"**

A: "Great question. XGBoost and LightGBM actually performed nearly identically to Random Forest—R-squared of 0.632 and 0.635 versus 0.640. The difference is marginal. I chose Random Forest because it's more interpretable, has fewer hyperparameters to tune, and the built-in feature importance works seamlessly with SHAP. For a 1% performance difference, I prioritized explainability."

**Q: "How did you choose the 60/20/20 split ratio?"**

A: "60/20/20 is a standard split for datasets over 10,000 samples. It gives us enough training data for the model to learn, enough validation data to tune hyperparameters reliably, and enough test data to get a trustworthy performance estimate. With 12,534 properties, that means 7,500+ for training, which is sufficient for Random Forest convergence."

**Q: "What about properties outside your training distribution—new markets or unusual property types?"**

A: "Excellent question. The model will struggle with out-of-distribution data. If you feed it a property from a market it's never seen, or a property type with unique characteristics, predictions will be less reliable. That's why the Streamlit app includes confidence intervals—wider intervals signal higher uncertainty. For production use, I'd recommend flagging predictions where the input falls outside the training data range."

**Q: "How often would you need to retrain this model?"**

A: "Commercial real estate markets evolve, so I'd recommend quarterly retraining at minimum. You'd want to retrain sooner if there's a major market event—like an interest rate shock or economic recession—that shifts rental dynamics. The beauty of the pipeline we built is that retraining is fully automated—just load new data and run the script."

**Q: "What's the computational cost to run the Streamlit app?"**

A: "Very low. Single predictions take under 50 milliseconds. The entire model file is 38MB, and the app uses less than 500MB of RAM when running. You could host this on a $5/month cloud server. For batch predictions of 1,000 properties, it's under 10 seconds."

**Q: "Did you consider neural networks?"**

A: "I did explore neural networks early on, but for tabular data with 12,000 samples, tree-based models (Random Forest, XGBoost) generally outperform deep learning. Neural nets shine when you have massive datasets (100K+ samples) or unstructured data like images and text. For structured real estate data, Random Forest is the right tool."

**Q: "How do you handle categorical features with hundreds of levels—like city names?"**

A: "Target encoding with cross-validation. For high-cardinality categoricals like City or Submarket, I encoded each level by its mean target value—calculated using 5-fold CV to prevent overfitting. This creates a numerical feature the model can use while preserving the signal. It's far better than one-hot encoding, which would create hundreds of sparse binary features."

**Q: "What was the hardest part of this project?"**

A: "Honestly, data cleaning and leakage prevention. Feature engineering and modeling are fun—everyone wants to do that. But ensuring data quality and preventing leakage required discipline and paranoia. I spent a week just validating that there was zero address overlap between train and test sets. It's not glamorous, but it's what separates real projects from broken ones."

---

## General Delivery Tips

**Pacing:**
- Speak clearly and at moderate pace (not rushed)
- Pause after key points to let them sink in
- Watch the clock—aim for 12-15 minutes total

**Body Language:**
- Make eye contact with audience
- Use hand gestures to emphasize points
- Stand confidently (avoid fidgeting)

**Slide Interaction:**
- Point to specific elements as you reference them
- Don't read directly from slides—use them as visual aids
- If doing live demo, practice transitions beforehand

**Handling Nerves:**
- Take a deep breath before starting
- Remember you know this material better than anyone in the room
- If you forget something, pause, regroup, and continue (audience won't notice)

**Engagement:**
- Vary your tone (don't be monotone)
- Ask rhetorical questions to keep audience thinking
- Use analogies when explaining complex concepts

**Time Management:**
- If running long, skip slide details (not entire slides)
- Practice full presentation at least twice before the actual event
- Have a watch or timer visible

---

**Good luck! You've built something impressive—now show it off with confidence.**
