# FarmGuard AI — Streamlit Dashboard
# Farmer facing web interface for crop disease detection

import streamlit as st
import requests
from PIL import Image
import io

# CONFIGURATION
API_URL = "http://127.0.0.1:8000"

# PAGE SETUP
st.set_page_config(page_title="FarmGuard AI", page_icon="🌿", layout="centered")

# HEADER
st.title("🌿 FarmGuard AI")
st.subheader(
    "Crop Disease Detection and Treatment Recommendations for Nigerian Farmers"
)
st.markdown("---")


# API HEALTH CHECK
# Checks if the FastAPI backend is running
def check_api():
    response = requests.get(f"{API_URL}/health", timeout=5)
    response: bool 
    try:
        response = response.status_code == 200
        return response
    except Exception as e:
        st.error(f"Error checking API status: {e}")
        return False


# SIDEBAR
with st.sidebar:
    st.header("ℹ️ About FarmGuard AI")
    st.write("""
    FarmGuard AI helps Nigerian farmers detect 
    crop diseases from leaf photos instantly and gives out treatment recommendations.
    """)

    st.markdown("---")
    st.header("🌱 Supported Crops")
    st.write("✅ Tomato")
    st.write("✅ Pepper")
    st.write("✅ Potato")

    st.markdown("---")
    st.header("🔍 Detectable Diseases")
    st.write("• Tomato Early Blight")
    st.write("• Pepper Bacterial Spot")
    st.write("• Potato Early Blight")
    st.write("• Healthy crops detected too!")

    st.markdown("---")

    # API status indicator
    st.header("⚙️ System Status")
    if check_api():
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline — Start the FastAPI server")

# MAIN CONTENT
st.header("📸 Upload a Leaf Photo")
st.write("Take a clear photo of a crop leaf and upload it below.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear photo of a single crop leaf",
)

# PREDICTION
if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    # Analyze button
    st.markdown("---")
    if st.button("🔍 Analyze Leaf", use_container_width=True):
        with st.spinner("Analyzing your crop leaf..."):
            try:
                # Send image to FastAPI
                uploaded_file.seek(0)
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                }
                response = requests.post(f"{API_URL}/predict", files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()

                    st.markdown("---")
                    st.header("📊 Analysis Results")

                    # Display results based on healthy or diseased
                    if result["is_healthy"]:
                        st.success(f"✅ {result['disease']}")
                        st.balloons()
                    else:
                        if result["severity"] == "Severe":
                            st.error(f"🚨 {result['disease']}")
                        elif result["severity"] == "Moderate":
                            st.warning(f"⚠️ {result['disease']}")
                        else:
                            st.info(f"ℹ️ {result['disease']}")

                    # Metrics row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Confidence", result["confidence"])
                    with col2:
                        st.metric("⚠️ Severity", result["severity"])
                    with col3:
                        status = "Healthy ✅" if result["is_healthy"] else "Diseased ❌"
                        st.metric("🌱 Status", status)

                    # AI Recommendation
                    st.markdown("---")
                    st.subheader("💊 AI Treatment Recommendation")
                    st.write(result["recommendation"])

                    # Warning for severe cases
                    if not result["is_healthy"] and result["severity"] == "Severe":
                        st.error("""
                        ⚠️ URGENT: This is a severe infection.
                        Act immediately to prevent large crop losses.
                        """)

                else:
                    st.error(
                        f"API Error: {response.json().get('detail', 'Unknown error')}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to API. Make sure the FastAPI server is running."
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# FOOTER
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "FarmGuard AI — Helping Nigerian Farmers Protect Their Crops"
    "</div>",
    unsafe_allow_html=True,
)
