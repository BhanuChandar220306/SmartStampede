def generate_recommendation(
    overall_risk,
    highest_risk_zone,
    density,
    movement,
    cpi
):

    # ==========================================
    # CRITICAL
    # ==========================================

    if overall_risk == "CRITICAL":

        reasons = []

        if cpi >= 75:
            reasons.append(
                f"very high crowd pressure (CPI {cpi:.1f})"
            )

        if density >= 90:
            reasons.append(
                f"critical density ({density:.1f}%)"
            )

        if movement >= 75:
            reasons.append(
                f"high crowd movement ({movement:.1f})"
            )

        reason_text = ", ".join(reasons)

        return (
            "🚨 **CRITICAL CROWD CONDITION**\n\n"

            f"📍 **Affected Zone:** {highest_risk_zone}\n\n"

            f"💥 **Crowd Pressure Index:** {cpi:.1f}\n\n"

            f"📊 **Density:** {density:.1f}%\n\n"

            f"🔄 **Movement:** {movement:.1f}\n\n"

            f"**Detected Factors:** {reason_text}\n\n"

            "### 🚨 Immediate Actions\n\n"

            "• Immediately control incoming crowd flow.\n\n"

            f"• Redirect people away from {highest_risk_zone}.\n\n"

            "• Alert security personnel immediately.\n\n"

            "• Keep emergency exits clear.\n\n"

            "• Prevent additional people from entering the affected zone.\n\n"

            "• Continuously monitor the zone."
        )


    # ==========================================
    # HIGH
    # ==========================================

    elif overall_risk == "HIGH":

        reasons = []

        if cpi >= 55:
            reasons.append(
                f"elevated crowd pressure (CPI {cpi:.1f})"
            )

        if density >= 70:
            reasons.append(
                f"high density ({density:.1f}%)"
            )

        if movement >= 55:
            reasons.append(
                f"increased movement ({movement:.1f})"
            )

        reason_text = ", ".join(reasons)

        return (
            "⚠️ **HIGH CROWD RISK**\n\n"

            f"📍 **Affected Zone:** {highest_risk_zone}\n\n"

            f"💥 **Crowd Pressure Index:** {cpi:.1f}\n\n"

            f"📊 **Density:** {density:.1f}%\n\n"

            f"🔄 **Movement:** {movement:.1f}\n\n"

            f"**Detected Factors:** {reason_text}\n\n"

            "### ⚠️ Recommended Actions\n\n"

            "• Reduce incoming crowd flow.\n\n"

            f"• Redirect people away from {highest_risk_zone}.\n\n"

            "• Increase security monitoring.\n\n"

            "• Keep entry and exit routes clear.\n\n"

            "• Prepare emergency response personnel."
        )


    # ==========================================
    # MEDIUM
    # ==========================================

    elif overall_risk == "MEDIUM":

        reasons = []

        if cpi >= 35:
            reasons.append(
                f"moderate crowd pressure (CPI {cpi:.1f})"
            )

        if density >= 40:
            reasons.append(
                f"moderate density ({density:.1f}%)"
            )

        if movement >= 35:
            reasons.append(
                f"increased movement ({movement:.1f})"
            )

        reason_text = ", ".join(reasons)

        return (
            "🟡 **MODERATE CROWD LEVEL**\n\n"

            f"📍 **Zone:** {highest_risk_zone}\n\n"

            f"💥 **Crowd Pressure Index:** {cpi:.1f}\n\n"

            f"📊 **Density:** {density:.1f}%\n\n"

            f"🔄 **Movement:** {movement:.1f}\n\n"

            f"**Detected Factors:** {reason_text}\n\n"

            "### 🟡 Recommended Actions\n\n"

            "• Continue monitoring the zone.\n\n"

            "• Watch for rapid crowd movement.\n\n"

            "• Keep entry and exit routes clear.\n\n"

            "• Be prepared to control incoming crowd flow."
        )


    # ==========================================
    # LOW
    # ==========================================

    else:

        return (
            "🟢 **CROWD CONDITION NORMAL**\n\n"

            f"📍 **Most Active Zone:** {highest_risk_zone}\n\n"

            f"💥 **Crowd Pressure Index:** {cpi:.1f}\n\n"

            f"📊 **Density:** {density:.1f}%\n\n"

            f"🔄 **Movement:** {movement:.1f}\n\n"

            "### ✅ Current Status\n\n"

            "• Crowd pressure is currently within the normal range.\n\n"

            "• Continue regular monitoring.\n\n"

            "• Keep emergency routes accessible.\n\n"

            "• No immediate intervention is indicated."
        )