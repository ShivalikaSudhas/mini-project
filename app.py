"""
app.py — Interactive Document Forgery Detection Terminal App
Enter an image path or press Enter to pick from presets.
Tells you whether the document is GENUINE or FORGED with confidence percentage.
"""

import sys
import os
from pathlib import Path
from src.visual_model import predict_visual

PRESETS = {
    "1": ("Aadhaar (Genuine)", r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\aadhaar\genuine\0000.jpg"),
    "2": ("Aadhaar (Forged)",  r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\aadhaar\forged\0000.jpg"),
    "3": ("PAN Card (Genuine)", r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\english_hindi\pan\genuine\0000.jpg"),
    "4": ("PAN Card (Forged)",  r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\english_hindi\pan\forged\0000.jpg"),
    "5": ("Driving Licence (Genuine)", r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\driving_licence\genuine\0000.jpg"),
    "6": ("Driving Licence (Forged)",  r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\driving_licence\forged\0000.jpg"),
    "7": ("Voter ID (Genuine)", r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\voter_id\genuine\0000.jpg"),
    "8": ("Voter ID (Forged)",  r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\voter_id\forged\0000.jpg"),
    "9": ("Passport (Genuine)", r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\english_hindi\passport\genuine\0000.jpg"),
    "10": ("Passport (Forged)", r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\english_hindi\passport\forged\0000.jpg"),
}

def analyze_image(img_path_str: str):
    p = Path(img_path_str.strip().strip('"').strip("'"))
    if not p.is_file():
        print(f"\n❌ Error: File not found -> {p}\n")
        return

    print(f"\n🔍 Analyzing: {p.name}")
    print("⏳ Running EfficientNet-B0 Visual Model...")

    res = predict_visual(str(p))
    pred = res["prediction"].upper()
    prob_forged = res["probability_forged"] * 100
    prob_genuine = res["probability_genuine"] * 100

    print("\n" + "=" * 50)
    if pred == "FORGED":
        print(f"🚨 RESULT: [ FORGED DOCUMENT DETECTED ]")
    else:
        print(f"✅ RESULT: [ GENUINE DOCUMENT ]")
    print("=" * 50)
    print(f"• Confidence (Forged)  : {prob_forged:6.2f}%")
    print(f"• Confidence (Genuine) : {prob_genuine:6.2f}%")
    print(f"• Raw Classification   : {pred}")
    print("=" * 50 + "\n")

def main():
    print("\n" + "#" * 60)
    print("  IDENTITY DOCUMENT FORGERY DETECTION SYSTEM")
    print("  Backbone: EfficientNet-B0 (Trained Checkpoint)")
    print("#" * 60)

    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in PRESETS:
            _, chosen_path = PRESETS[arg]
            analyze_image(chosen_path)
        else:
            analyze_image(arg)
        return

    while True:
        print("\n--- TEST PRESETS ---")
        for k, (label, path) in PRESETS.items():
            print(f" [{k:>2}] {label}")
        print(" [ q] Quit")
        print("--------------------")

        user_input = input("\nEnter preset number (1-10) OR paste any image path: ").strip()

        if user_input.lower() in ("q", "quit", "exit"):
            print("Exiting. Bye!")
            break

        if user_input in PRESETS:
            _, chosen_path = PRESETS[user_input]
            analyze_image(chosen_path)
        elif user_input:
            analyze_image(user_input)

if __name__ == "__main__":
    main()
