import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_feature_correlation import page_feature_correlation_body
from app_pages.page_predict_price import page_predict_price_body
from app_pages.page_ml_model import page_ml_model_body

# Create an instance of the app
app = MultiPage(app_name="House Prices Analysis")

# Add pages
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Feature Correlation", page_feature_correlation_body)
app.add_page("Predict House Price", page_predict_price_body)
app.add_page("ML Model", page_ml_model_body)

# Run the app
app.run()
