import os
import time
import wave

import numpy as np
import pygame
from gtts import gTTS


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_DIR = "alert_audio"

os.makedirs(
    AUDIO_DIR,
    exist_ok=True
)


# ============================================================
# INITIALIZE AUDIO
# ============================================================

pygame.mixer.init()


# ============================================================
# SIREN
# ============================================================

SIREN_FILE = os.path.join(
    AUDIO_DIR,
    "emergency_siren.wav"
)


def generate_siren(
    filename,
    duration=5,
    sample_rate=44100
):

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    frequency = (
        650
        +
        500 *
        (
            0.5
            +
            0.5 *
            np.sin(
                2 *
                np.pi *
                0.8 *
                t
            )
        )
    )

    phase = (
        2 *
        np.pi *
        np.cumsum(
            frequency
        )
        /
        sample_rate
    )

    audio = (
        0.85 *
        np.sin(
            phase
        )
    )

    audio += (
        0.25 *
        np.sin(
            2 *
            phase
        )
    )

    audio = (
        audio /
        np.max(
            np.abs(audio)
        )
    )

    audio = (
        audio *
        32767
    ).astype(
        np.int16
    )

    with wave.open(
        filename,
        "wb"
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(
            audio.tobytes()
        )


if not os.path.exists(
    SIREN_FILE
):

    print(
        "Creating emergency siren..."
    )

    generate_siren(
        SIREN_FILE
    )


# ============================================================
# PLAY AUDIO
# ============================================================

def play_audio(
    filename
):

    print(
        "Playing:",
        filename
    )

    try:

        pygame.mixer.music.load(
            filename
        )

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            time.sleep(
                0.1
            )

        print(
            "Finished:",
            filename
        )

        return True

    except Exception as e:

        print(
            "AUDIO PLAY ERROR:",
            e
        )

        return False


# ============================================================
# CREATE TTS
# ============================================================

def create_tts(
    filename,
    text,
    language
):

    # --------------------------------------------------------
    # If already generated, don't generate again.
    # --------------------------------------------------------

    if os.path.exists(
        filename
    ):

        print(
            "Using existing:",
            filename
        )

        return True


    print(
        f"Creating {language} voice..."
    )

    try:

        tts = gTTS(
            text=text,
            lang=language,
            slow=False
        )

        tts.save(
            filename
        )

        print(
            "Created:",
            filename
        )

        return True

    except Exception as e:

        print(
            f"TTS ERROR ({language}):",
            e
        )

        return False


# ============================================================
# FULL ALERT
# ============================================================

def play_alert_sequence(
    result
):

    try:

        # ====================================================
        # GET INFORMATION
        # ====================================================

        risk = result.get(
            "risk_level",
            "ALERT"
        )

        danger_sector = result.get(
            "danger_sector",
            "UNKNOWN"
        )

        yolo = int(
            result.get(
                "yolo_count",
                0
            )
        )

        csrnet = float(
            result.get(
                "csrnet_count",
                0
            )
        )

        hybrid = float(
            result.get(
                "hybrid_count",
                0
            )
        )

        movement = float(
            result.get(
                "movement_score",
                0
            )
        )

        cpi = float(
            result.get(
                "cpi",
                0
            )
        )


        print()
        print(
            "🚨 AUDIO ALERT STARTED"
        )


        # ====================================================
        # 1. FIRST SIREN
        # ====================================================

        print(
            "1/5 - SIREN"
        )

        play_audio(
            SIREN_FILE
        )


        # ====================================================
        # TEXT
        # ====================================================

        english_text = f"""
Emergency crowd alert.

Attention everyone.

A dangerous crowd condition has been detected.

Danger sector is {danger_sector}.

Risk level is {risk}.

The estimated crowd size is {hybrid:.0f} people.

YOLO detected {yolo} people.

CSRNet estimated {csrnet:.1f} people.

Movement score is {movement:.1f}.

Crowd pressure index is {cpi:.1f}.

Immediate crowd control action is required.

Please move away from sector {danger_sector}.

Security personnel should respond immediately.
"""


        telugu_text = f"""
అత్యవసర జనసమూహ హెచ్చరిక.

అందరూ జాగ్రత్తగా వినండి.

ప్రమాదకరమైన జనసమూహ పరిస్థితి గుర్తించబడింది.

ప్రమాద ప్రాంతం {danger_sector}.

రిస్క్ స్థాయి {risk}.

అంచనా ప్రకారం జనసమూహంలో {hybrid:.0f} మంది ఉన్నారు.

వెంటనే జనసమూహ నియంత్రణ చర్యలు తీసుకోండి.

ప్రమాద ప్రాంతం {danger_sector} నుండి దూరంగా వెళ్లండి.

భద్రతా సిబ్బంది వెంటనే స్పందించాలి.
"""


        hindi_text = f"""
आपातकालीन भीड़ चेतावनी।

सभी लोग ध्यान दें।

एक खतरनाक भीड़ की स्थिति का पता चला है।

खतरनाक क्षेत्र {danger_sector} है।

जोखिम स्तर {risk} है।

अनुमान के अनुसार भीड़ में {hybrid:.0f} लोग हैं।

तुरंत भीड़ नियंत्रण की कार्रवाई आवश्यक है।

कृपया खतरनाक क्षेत्र {danger_sector} से दूर जाएं।

सुरक्षा कर्मियों को तुरंत कार्रवाई करनी चाहिए।
"""


        # ====================================================
        # FILES
        # ====================================================

        english_file = os.path.join(
            AUDIO_DIR,
            "english_alert.mp3"
        )

        telugu_file = os.path.join(
            AUDIO_DIR,
            "telugu_alert.mp3"
        )

        hindi_file = os.path.join(
            AUDIO_DIR,
            "hindi_alert.mp3"
        )


        # ====================================================
        # 2. ENGLISH
        # ====================================================

        print(
            "2/5 - ENGLISH VOICE"
        )

        if create_tts(
            english_file,
            english_text,
            "en"
        ):

            play_audio(
                english_file
            )


        # ====================================================
        # 3. TELUGU
        # ====================================================

        print(
            "3/5 - TELUGU VOICE"
        )

        if create_tts(
            telugu_file,
            telugu_text,
            "te"
        ):

            play_audio(
                telugu_file
            )


        # ====================================================
        # 4. HINDI
        # ====================================================

        print(
            "4/5 - HINDI VOICE"
        )

        if create_tts(
            hindi_file,
            hindi_text,
            "hi"
        ):

            play_audio(
                hindi_file
            )


        # ====================================================
        # 5. FINAL SIREN
        # ====================================================

        print(
            "5/5 - FINAL SIREN"
        )

        play_audio(
            SIREN_FILE
        )


        print()
        print(
            "🚨 AUDIO ALERT FINISHED"
        )


    except Exception as e:

        print()
        print(
            "=========================================="
        )

        print(
            " AUDIO ALERT FAILED"
        )

        print(
            "Error:",
            e
        )

        print(
            "=========================================="
        )