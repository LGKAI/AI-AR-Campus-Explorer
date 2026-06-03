import cv2

def draw_result(img, box, label):
    x1, y1, x2, y2 = box
    color = (0, 255, 0) if "UNKNOWN" not in label else (0, 0, 255)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img
