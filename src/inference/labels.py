CLASS_LABELS = [
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


def display_label(label: str) -> str:
    plant, disease = label.split("___", 1)
    return f"{plant} - {disease.replace('_', ' ')}"
