import streamlit as st
import cv2
import time

from ai_engine import process_frame
from recommendation import generate_recommendation


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Event Monitoring System",
    page_icon="🚨",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("🚨 Smart Event Monitoring System")

st.caption(
    "AI-Based Crowd Monitoring and Stampede Prevention"
)

st.divider()


# ==========================================
# TOP METRICS
# ==========================================

metric1 = st.empty()
metric2 = st.empty()
metric3 = st.empty()
metric4 = st.empty()
metric5 = st.empty()
metric6 = st.empty()

st.divider()


# ==========================================
# CCTV + ALERT
# ==========================================

left, right = st.columns([2, 1])


# ==========================================
# CCTV
# ==========================================

with left:

    st.subheader("📹 Live CCTV Feed")

    video_placeholder = st.empty()


# ==========================================
# ALERT
# ==========================================

with right:

    st.subheader("🚨 Alert Panel")

    alert_placeholder = st.empty()

    st.write("### 🧠 Recommended Action")

    recommendation_placeholder = st.empty()


st.divider()


# ==========================================
# ZONE MONITORING
# ==========================================

st.subheader("📊 Zone Monitoring")

zone1, zone2, zone3 = st.columns(3)


with zone1:

    st.markdown("### 🟢 ZONE A")

    zone_a_placeholder = st.empty()


with zone2:

    st.markdown("### 🟡 ZONE B")

    zone_b_placeholder = st.empty()


with zone3:

    st.markdown("### 🔴 ZONE C")

    zone_c_placeholder = st.empty()


st.divider()


# ==========================================
# VIDEO
# ==========================================

video_path = "videos/crowd12.mp4"
#video_path="C:/Users/bhanu/Downloads/बेंगलुरु में RCB की जीत के जश्न में मची भगदड़ _ bangalore stampede _ RCB _ Virat Kohli.mp4"

cap = cv2.VideoCapture(video_path)


if not cap.isOpened():

    st.error(
        "Unable to open CCTV video.\n\n"
        "Check that the video path is correct."
    )

    st.stop()


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ======================================
    # AI PROCESSING
    # ======================================

    data = process_frame(frame)



    # ======================================
    # GET RESULTS
    # ======================================

    processed_frame = data["frame"]

    total_people = round(data["hybrid_count"])

    zone_a_count = data["zone_a_count"]
    zone_b_count = data["zone_b_count"]
    zone_c_count = data["zone_c_count"]

    density_a = data["density_a"]
    density_b = data["density_b"]
    density_c = data["density_c"]

    risk_a = data["risk_a"]
    risk_b = data["risk_b"]
    risk_c = data["risk_c"]

    cpi_a = data["cpi_a"]
    cpi_b = data["cpi_b"]
    cpi_c = data["cpi_c"]

    movement = data["movement"]

    highest_risk_zone = data["highest_risk_zone"]

    highest_density = data["highest_density"]

    highest_cpi = data["highest_cpi"]

    overall_risk = data["overall_risk"]


    # ======================================
    # DISPLAY VIDEO
    # ======================================

    processed_frame = cv2.cvtColor(
        processed_frame,
        cv2.COLOR_BGR2RGB
    )

    video_placeholder.image(
        processed_frame,
        channels="RGB"
    )


    # ======================================
    # TOP METRICS
    # ======================================

    metric1.metric(
        "👥 Total People",
        total_people
    )

    metric2.metric(
        "📍 Danger Zone",
        highest_risk_zone
    )

    metric3.metric(
        "📊 Highest Density",
        f"{highest_density:.1f}%"
    )

    metric4.metric(
        "🔄 Movement",
        f"{movement:.1f}"
    )

    metric5.metric(
    "🤖 CSRNet Estimate",
    f"{data['csrnet_count']:.1f}"
)

    metric6.metric(
    "🔀 Hybrid Estimate",
    f"{data['hybrid_count']:.1f}"
)


    # ======================================
    # INTELLIGENT RECOMMENDATION
    # ======================================

    recommendation = generate_recommendation(
        overall_risk,
        highest_risk_zone,
        highest_density,
        movement,
        highest_cpi
    )


    # ======================================
    # ALERT DISPLAY
    # ======================================

    if overall_risk == "CRITICAL":

        alert_placeholder.error(
            f"🚨 CRITICAL CROWD PRESSURE\n\n"
            f"📍 {highest_risk_zone}\n\n"
            f"💥 CPI: {highest_cpi:.1f}\n\n"
            f"🔄 Movement: {movement:.1f}"
        )


    elif overall_risk == "HIGH":

        alert_placeholder.warning(
            f"⚠️ HIGH CROWD PRESSURE\n\n"
            f"📍 {highest_risk_zone}\n\n"
            f"💥 CPI: {highest_cpi:.1f}\n\n"
            f"🔄 Movement: {movement:.1f}"
        )


    elif overall_risk == "MEDIUM":

        alert_placeholder.warning(
            f"🟡 MODERATE CROWD PRESSURE\n\n"
            f"📍 Monitor {highest_risk_zone}\n\n"
            f"💥 CPI: {highest_cpi:.1f}"
        )


    else:

        alert_placeholder.success(
            "🟢 CROWD CONDITION NORMAL"
        )


    # ======================================
    # RECOMMENDATION
    # ======================================

    recommendation_placeholder.markdown(
        recommendation
    )


    # ======================================
    # ZONE A
    # ======================================

    zone_a_placeholder.markdown(
        f"""
        ### 🟢 ZONE A

        **👥 People:** {zone_a_count}

        **📊 Density:** {density_a:.1f}%

        **💥 CPI:** {cpi_a:.1f}

        **🔄 Movement:** {movement:.1f}

        **⚠️ Risk:** {risk_a}
        """
    )


    # ======================================
    # ZONE B
    # ======================================

    zone_b_placeholder.markdown(
        f"""
        ### 🟡 ZONE B

        **👥 People:** {zone_b_count}

        **📊 Density:** {density_b:.1f}%

        **💥 CPI:** {cpi_b:.1f}

        **🔄 Movement:** {movement:.1f}

        **⚠️ Risk:** {risk_b}
        """
    )


    # ======================================
    # ZONE C
    # ======================================

    zone_c_placeholder.markdown(
        f"""
        ### 🔴 ZONE C

        **👥 People:** {zone_c_count}

        **📊 Density:** {density_c:.1f}%

        **💥 CPI:** {cpi_c:.1f}

        **🔄 Movement:** {movement:.1f}

        **⚠️ Risk:** {risk_c}
        """
    )


    # ======================================
    # VIDEO SPEED
    # ======================================

    time.sleep(0.08)


# ==========================================
# RELEASE VIDEO
# ==========================================

cap.release()

st.success(
    "CCTV video processing completed."
)