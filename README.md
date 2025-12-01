# ai-sud-modeling-impact
End-to-end analytics project predicting citation impact and recommending datasets for AI-driven Substance Use Disorder (SUD) research using metadata from 90+ papers. Includes data cleaning, NLP, regression &amp; classification models, and SHAP-based interpretability.
This project builds an end-to-end analytics pipeline to understand what drives high-impact AI research in the Substance Use Disorder (SUD) domain. Using a curated corpus of SUD-related AI papers, I:

Cleaned and standardized metadata (citations, impact factor, dataset, ethics, methods, etc.).

Performed exploratory analysis, correlation heatmaps, and basic NLP on methods and titles.

Trained regression models (Multiple Linear Regression, Random Forest, XGBoost) to predict citation counts, and classification models to distinguish highly-cited vs less-cited work.

Used SHAP values and feature importance to explain which factors (journal impact factor, dataset type, ethics statements) most strongly influence citation impact.

The result is a modeling and recommendation framework that helps researchers select datasets and framing strategies more likely to lead to high-impact, ethically aligned AI-SUD research.
