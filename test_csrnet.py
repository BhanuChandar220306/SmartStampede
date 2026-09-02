import cv2
import torch
from torchvision import transforms

from csrnet import CSRNet


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading CSRNet...")

model = CSRNet()

checkpoint = torch.load(
    "weights.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint)

model.eval()

print("✅ CSRNet loaded")


# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize(
        (512, 512)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==========================================
# OPEN VIDEO
# ==========================================

video_path = "videos/crowd.mp4"

cap = cv2.VideoCapture(
    video_path
)


if not cap.isOpened():

    print("❌ Unable to open video")

    exit()


# ==========================================
# READ FIRST FRAME
# ==========================================

ret, frame = cap.read()


if not ret:

    print("❌ Unable to read frame")

    cap.release()

    exit()


print(
    "Frame size:",
    frame.shape
)


# ==========================================
# PREPARE FRAME
# ==========================================

rgb_frame = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
)

input_tensor = transform(
    rgb_frame
)

input_tensor = input_tensor.unsqueeze(0)


# ==========================================
# CSRNET INFERENCE
# ==========================================

print("Running CSRNet...")

with torch.no_grad():

    density_map = model(
        input_tensor
    )


# ==========================================
# COUNT PEOPLE
# ==========================================

count = density_map.sum().item()


# ==========================================
# RESULTS
# ==========================================

print()
print("==============================")
print("CSRNet Crowd Estimate")
print("==============================")
print(
    f"Estimated People: {count:.2f}"
)
print(
    f"Density Map Shape: {density_map.shape}"
)
print("==============================")


cap.release()