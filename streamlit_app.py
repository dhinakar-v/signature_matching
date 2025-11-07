import base64
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from PIL import Image

# Load environment variables
load_dotenv()

# Page configuration (no sidebar)
st.set_page_config(page_title="Signature Comparison Tool", page_icon="✍️", layout="wide")

# --- Header ---
st.title("✍️ Signature Comparison Tool")
st.markdown(
    """
Upload **two signature images** to compare their similarity using AI.
The tool will provide a similarity score and a detailed visual analysis.
"""
)

# --- Fetch Azure Credentials from .env ---
azure_endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("SECRET_KEY")
deployment_name = os.getenv("DEPLOYMENT")
api_version = os.getenv("VERSION")


# --- Function to encode images in base64 ---
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# --- Upload Areas ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Signature 1")
    signature1 = st.file_uploader(
        "Upload the first signature", type=["png", "jpg", "jpeg"], key="sig1"
    )
    if signature1:
        img1 = Image.open(signature1)
        st.image(img1, caption="Signature 1", use_container_width=True)

with col2:
    st.subheader("📝 Signature 2")
    signature2 = st.file_uploader(
        "Upload the second signature", type=["png", "jpg", "jpeg"], key="sig2"
    )
    if signature2:
        img2 = Image.open(signature2)
        st.image(img2, caption="Signature 2", use_container_width=True)

# --- Compare Button ---
st.markdown("---")
if st.button("🔍 Compare Signatures", type="primary", use_container_width=True):
    if not azure_endpoint or not api_key:
        st.error("⚠️ Azure OpenAI credentials missing in your `.env` file.")
    elif not signature1 or not signature2:
        st.error("⚠️ Please upload both signature images.")
    else:
        try:
            with st.spinner("Analyzing signatures... Please wait."):
                # Initialize model
                llm = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    deployment_name=deployment_name,
                    api_version=api_version,
                )

                # Encode images
                base64_image1 = encode_image(Image.open(signature1))
                base64_image2 = encode_image(Image.open(signature2))

                # Formatted prompt
                prompt = """
You are an expert in signature verification and forensic handwriting analysis.

Analyze the two uploaded signature images and respond in **structured Markdown format** with the following sections:

### 🧮 Similarity Score
Provide a **score between 0 and 100**, where:
- 90–100 → Nearly identical (same person)
- 70–89 → Very similar (likely same person)
- 50–69 → Moderately similar (possibly same person)
- 30–49 → Some resemblance but notable differences
- 0–29 → Very different (likely different people)

### ✍️ Detailed Comparison
Compare and describe:
- Overall shape and flow  
- Letter formation and style  
- Slant and angle consistency  
- Pressure and line thickness  
- Spacing and proportion  
- Unique features or flourishes

### ⚠️ Observed Differences or Issues
List specific discrepancies or anomalies such as:
- Hesitation, tremors, or uneven flow  
- Different slants or stroke directions  
- Extra/missing loops or strokes  
- Size or spacing inconsistencies  
- Potential signs of forgery

Make sure your answer is **clearly formatted** in Markdown and uses bullet points and section headers.
"""

                # Create chat message
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image1}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image2}"
                            },
                        },
                    ]
                )

                # Get response
                response = llm.invoke([message])

                # --- Display results ---
                st.success("✅ Analysis Complete!")
                st.markdown("### 📊 AI-Generated Signature Comparison Report")
                st.markdown(response.content, unsafe_allow_html=True)

                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=response.content,
                    file_name="signature_comparison_report.md",
                    mime="text/markdown",
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info(
                "Ensure your Azure GPT-4 Vision deployment is active and credentials are correct."
            )

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Azure OpenAI GPT-4 Vision • Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)
