from src.inference.labels import CLASS_LABELS, display_label


def test_class_labels_match_training_order() -> None:
    assert CLASS_LABELS == [
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


def test_display_label_is_readable() -> None:
    assert display_label("Apple___Black_rot") == "Apple - Black rot"
    assert display_label("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)") == (
        "Grape - Leaf blight (Isariopsis Leaf Spot)"
    )
