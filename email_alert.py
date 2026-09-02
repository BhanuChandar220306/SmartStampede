import smtplib
import ssl
from email.message import EmailMessage


# ============================================================
# GMAIL CONFIGURATION
# ============================================================

SENDER_EMAIL = "stampedmanagement5@gmail.com"

# Use a Google App Password here.
# DO NOT use your normal Gmail password.
GMAIL_APP_PASSWORD = "hqtrmtcbymxjtglm"


# Add 1 to 5 recipients.
ALERT_EMAILS = [
    "bhanuchandarchagantipati@gmail.com",
    "abhiab77777@gmail.com",
    "knharikachithajallu@gmail.com ",
    "surekhautla1205@gmail.com ",
    "viswarupavaddepalli12@gmail.com ",
    "dhanu.sunny54@gmail.com "
]


# ============================================================
# SEND ALERT EMAIL
# ============================================================

def send_alert_email(result):
    """
    Sends one crowd ALERT email to all configured recipients.
    """

    risk = result.get("risk_level", "UNKNOWN")
    danger_sector = result.get("danger_sector", "NONE")

    hybrid_count = float(
        result.get("hybrid_count", 0)
    )

    csrnet_count = float(
        result.get("csrnet_count", 0)
    )

    yolo_count = int(
        result.get("yolo_count", 0)
    )

    cpi = float(
        result.get("cpi", 0)
    )

    movement = float(
        result.get("movement_score", 0)
    )

    highest_density = float(
        result.get("highest_density", 0)
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if risk != "ALERT":
        return False

    if not ALERT_EMAILS:
        print("\nEMAIL ERROR: No recipient emails configured.")
        return False

    # --------------------------------------------------------
    # Email subject
    # --------------------------------------------------------

    subject = (
        f"🚨 CROWD ALERT - "
        f"Sector {danger_sector}"
    )

    # --------------------------------------------------------
    # Email body
    # --------------------------------------------------------

    body = f"""
🚨 SMART EVENT MONITORING SYSTEM
================================

IMMEDIATE CROWD ALERT

Risk Level       : {risk}
Danger Sector    : {danger_sector}

--------------------------------
AI CROWD ANALYSIS
--------------------------------

YOLO Estimate    : {yolo_count}
CSRNet Estimate  : {csrnet_count:.1f}
Hybrid Estimate  : {hybrid_count:.1f}

Movement Score   : {movement:.1f}
CPI              : {cpi:.1f}
Highest Density  : {highest_density:.1f}

--------------------------------

IMMEDIATE RESPONSE IS REQUIRED.

Please check the CCTV feed and take
appropriate crowd-control measures.

This alert was generated automatically
by the AI Crowd Monitoring System.
"""

    # --------------------------------------------------------
    # Create email
    # --------------------------------------------------------

    message = EmailMessage()

    message["From"] = SENDER_EMAIL
    message["To"] = ", ".join(ALERT_EMAILS)
    message["Subject"] = subject

    message.set_content(body)

    # --------------------------------------------------------
    # Gmail SMTP
    # --------------------------------------------------------

    context = ssl.create_default_context()

    try:

        print("\n==========================================")
        print(" SENDING CROWD ALERT EMAIL")
        print("==========================================")

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            context=context
        ) as server:

            server.login(
                SENDER_EMAIL,
                GMAIL_APP_PASSWORD
            )

            server.send_message(message)

        print(
            f"ALERT EMAIL SENT TO "
            f"{len(ALERT_EMAILS)} RECIPIENT(S)"
        )

        print(
            "Recipients:",
            ", ".join(ALERT_EMAILS)
        )

        print("==========================================\n")

        return True

    except Exception as e:

        print("\n==========================================")
        print(" EMAIL SENDING FAILED")
        print("==========================================")
        print("Error:", e)
        print("==========================================\n")

        return False