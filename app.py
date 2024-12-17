import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.data_summary import data_summary_body
from app_pages.feature_correlation import feature_correlation_body
from app_pages.predict_sale_price import predict_sale_price_body
from app_pages.hypotheses_validation import hypotheses_validation_body
from app_pages.model_performance import model_performance_body

# Create an instance of the app
app = MultiPage(app_name="House Prices Analysis")

# Add pages
app.add_page("Quick Project Summary", data_summary_body)
app.add_page("Feature Correlation", feature_correlation_body)
app.add_page("Predict House Price", predict_sale_price_body)
app.add_page("Hypotheses Validation", hypotheses_validation_body)
app.add_page("ML Model Summary", model_performance_body)

# Run the app
app.run()
