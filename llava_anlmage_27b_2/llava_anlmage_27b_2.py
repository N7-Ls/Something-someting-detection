import requests
import base64
from preprocess_head import crop_head

def image_to_base64(image_path: str) -> str:
    # Read the image file and convert it to base64 encoding
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

API_URL = "https://uncommutatively-unpersuadable-an.ngrok-free.dev/api/chat"
API_KEY = "upiceollama"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

# Change the filename to test different images
original_image = "pic/helmetA_3.jpg"
processed_image = crop_head(original_image)
img_b64 = image_to_base64(processed_image)

# 優化後的 Prompt：強調具體視覺特徵並引入推論步驟
new_prompt = (
    "You are inspecting whether a helmet chin strap is properly fastened. Follow each step carefully.\n\n"
    "Step 1 (Helmet Check): Is the person wearing a helmet? If No, output 'Result: No helmet' and stop.\n\n"
    "Step 2 (Chin Strap Inspection - Only if helmet is present):\n"
    "A chin strap runs from both sides of the helmet, down along the cheeks, and clips together under the chin.\n"
    "- FASTENED signs: a strap or band is visibly pulled snug across the underside of the chin, "
    "connecting left and right sides. The strap lies flat against the skin with no gap.\n"
    "- UNFASTENED signs: one or both straps hang loosely beside the face or dangle below the chin "
    "without touching the skin. A buckle or clip is visible hanging freely on the side of the face or neck.\n\n"
    "Step 3 (Decision):\n"
    "- If the strap is snug under the chin and connected on both sides → FASTENED\n"
    "- If the strap is loose, hanging, or not touching the chin → UNFASTENED\n"
    "- When in doubt, look for whether the buckle is clipped shut under the chin.\n\n"
    "Describe what you see under the chin and on the cheeks in one sentence, "
    "then on a new line output exactly one of:\n"
    "- 'Result: No helmet'\n"
    "- 'Result: Helmet on, strap fastened'\n"
    "- 'Result: Helmet on, strap unfastened'"
)

# 加入 options 設定 temperature 為 0，確保結果穩定一致
payload = {
    "model": "gemma3:27b",
    "messages": [
        {
            "role": "user",
            "content": new_prompt,
            "images": [img_b64]
        }
    ],
    "stream": False,
    "keep_alive": 0,
    "options": {
        "temperature": 0.0
    }
}

response = requests.post(API_URL, headers=headers, json=payload, timeout=300)

print("Status:", response.status_code)

response.raise_for_status()

resp_json = response.json()
answer = resp_json["message"]["content"]

print("\nAssistant answer:\n")
print(answer)