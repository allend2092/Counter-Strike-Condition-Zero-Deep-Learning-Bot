import cv2 as cv
import numpy as np
from time import time
from PIL import ImageGrab, Image
import pytesseract
import easyocr
import threading
import queue
import keyboard  # For detecting key presses

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=True)

def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
    return binary

def capture_screen(region=None):
    try:
        screenshot = ImageGrab.grab(bbox=region)
        screenshot_np_array = np.array(screenshot)
        screenshot_bgr = cv.cvtColor(screenshot_np_array, cv.COLOR_RGB2BGR)
        preprocessed = preprocess_image(screenshot_bgr)
        return screenshot, screenshot_bgr, preprocessed
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None, None, None

def read_text_from_region_esy(image, region, debug=False):
    cropped_image = image.crop(region)
    if debug:
        cropped_image.show()
    results = reader.readtext(np.array(cropped_image))
    text = ' '.join([result[1] for result in results])
    return text

def ocr_thread_function(image_queue):
    while True:
        try:
            image, preprocessed_image, region = image_queue.get(timeout=1)
            text = read_text_from_region_esy(preprocessed_image, region)
            print(text)
            image_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in OCR thread: {e}")

image_queue = queue.Queue()
ocr_thread = threading.Thread(target=ocr_thread_function, args=(image_queue,))
ocr_thread.start()

def calculate_fps(loop_time):
    current_time = time()
    fps = 1 / (current_time - loop_time)
    return fps, current_time

def main():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    loop_time = time()
    fps_accumulator = []
    fps_average_period = 10

    try:
        while True:
            pil_image, screenshot, preprocessed = capture_screen(region=(1120, 25, 1920, 625))
            if screenshot is not None:
                ocr_region = (40, 560, 100, 598)
                cv.rectangle(screenshot, (ocr_region[0], ocr_region[1]), (ocr_region[2], ocr_region[3]), (0, 0, 255), 2)
                cv.rectangle(preprocessed, (ocr_region[0], ocr_region[1]), (ocr_region[2], ocr_region[3]), (255, 255, 255), 2)
                cv.imshow('captured screen region', screenshot)
                cv.imshow('preprocessed region', preprocessed)
                image_queue.put((pil_image, Image.fromarray(preprocessed), ocr_region))

            fps, loop_time = calculate_fps(loop_time)
            fps_accumulator.append(fps)

            if len(fps_accumulator) == fps_average_period:
                avg_fps = sum(fps_accumulator) / len(fps_accumulator)
                print(f'Average FPS: {avg_fps:.2f}')
                fps_accumulator = []

            if cv.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        global ocr_thread_running
        ocr_thread_running = False
        ocr_thread.join()
        cv.destroyAllWindows()
        print('Done.')

if __name__ == "__main__":
    main()
