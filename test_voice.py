from gtts import gTTS
import pygame
import time


print("Creating English voice...")

tts = gTTS(
    text="Emergency crowd alert. Please move away from the dangerous area.",
    lang="en",
    slow=False
)

tts.save("test_english.mp3")

print("Voice file created.")


pygame.mixer.init()

pygame.mixer.music.load(
    "test_english.mp3"
)

pygame.mixer.music.play()

print("Playing voice...")

while pygame.mixer.music.get_busy():

    time.sleep(0.1)

print("Voice finished.")