# Import necessary libraries
import cv2 as cv
import numpy as np
from time import time
import pyautogui
from PIL import ImageGrab, Image
import pytesseract
import threading
import queue
import easyocr

# Create an EasyOCR Reader instance
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if you don't want to use GPU

def read_text_from_region_esy(image, region, debug=False):
    """
    Reads text from a specified region in the image using EasyOCR.

    Args:
        image (PIL.Image): The image to read from.
        region (tuple): The region (left, upper, right, lower) to extract text from.
        debug (bool): If True, show the cropped image for debugging.

    Returns:
        str: The text read from the region.
    """
    # Crop the image to the specified region
    cropped_image = image.crop(region)

    # If debug is True, show the cropped image
    if debug:
        cropped_image.show()

    # Use EasyOCR to do OCR on the cropped image
    results = reader.readtext(np.array(cropped_image))

    # Extracting text from the results
    text = ' '.join([result[1] for result in results])

    return text

# Flag to control the OCR thread's execution
ocr_thread_running = True
def ocr_thread_function(image_queue):
    global ocr_thread_running
    while ocr_thread_running:
        try:
            # Get the next image from the queue
            image, region = image_queue.get(timeout=1)  # Timeout to allow checking the flag

            # Perform OCR
            text = read_text_from_region_esy(image, region)

            # Process the text or print it
            print(text)

            # Mark the task as done
            image_queue.task_done()
        except queue.Empty:
            # No new image, continue to the next iteration
            continue
        except Exception as e:
            print(f"Error in OCR thread: {e}")

# Create a queue to hold images for OCR
image_queue = queue.Queue()

# Create and start the OCR thread
ocr_thread = threading.Thread(target=ocr_thread_function, args=(image_queue,))
ocr_thread.start()

def read_text_from_region(image, region, debug=False):
    """
    Reads text from a specified region in the image.

    Args:
        image (PIL.Image): The image to read from.
        region (tuple): The region (left, upper, right, lower) to extract text from.
        debug (bool): If True, show the cropped image for debugging.

    Returns:
        str: The text read from the region.
    """
    # Crop the image to the specified region
    cropped_image = image.crop(region)

    # If debug is True, show the cropped image
    if debug:
        cropped_image.show()
        # Optionally, save the image to a file for closer inspection
        # cropped_image.save("debug_cropped_image.png")

    # Use Tesseract to do OCR on the cropped image
    text = pytesseract.image_to_string(cropped_image)

    return text

def capture_screen(region=None):
    """
    Captures a specific area of the screen.

    Args:
        region (tuple, optional): The region to capture (left, top, right, bottom).
                                  Captures the full screen if None. Defaults to None.

    Returns:
        tuple: A tuple containing the PIL Image object and the captured screen image in BGR format as a NumPy array.
    """
    try:
        # Capture a specific region of the screen, or the full screen if region is None
        screenshot = ImageGrab.grab(bbox=region)

        # Convert the screenshot to a NumPy array and then to BGR format
        screenshot_np_array = np.array(screenshot)
        screenshot_bgr = cv.cvtColor(screenshot_np_array, cv.COLOR_RGB2BGR)

        # Return both the PIL Image object and the NumPy array
        return screenshot, screenshot_bgr

    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None, None

def calculate_fps(loop_time):
    """
    Calculates the frames per second (FPS).

    Args:
        loop_time (float): The time when the last frame was processed.

    Returns:
        tuple: The current FPS and the current time.
    """
    # Get the current time
    current_time = time()

    # Calculate the time difference between the current frame and the last frame
    # Then calculate the FPS as the inverse of this time difference
    fps = 1 / (current_time - loop_time)

    # Return the calculated FPS and the current time
    return fps, current_time

def main():
    global ocr_thread_running
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path to where Tesseract is installed

    # Initialize the time for FPS calculation
    loop_time = time()

    # Initialize a list to store FPS values for averaging
    fps_accumulator = []

    # Define the number of frames to average for FPS calculation
    fps_average_period = 10

    try:
        while True:
            # Capture the screen: I've found these coodinates to be best for Half-Life at 800x600
            pil_image, screenshot = capture_screen(region=(1120, 25, 1920, 625))
            # screenshot = capture_screen(region=(706, 10, 758, 45))

            # If the screenshot is successfully captured
            if screenshot is not None:
                # Define the coordinates for the OCR region (top-left and bottom-right)
                ocr_region = (40, 560, 100, 598)  # Update this to your desired region
                # Draw a red rectangle around the OCR region
                # Note: OpenCV uses BGR color format
                cv.rectangle(screenshot, (ocr_region[0], ocr_region[1]), (ocr_region[2], ocr_region[3]), (0, 0, 255), 2)

                # Display the screenshot in a window named 'Half-Life'
                cv.imshow('captured screen region', screenshot)

                # this was the previous implementation of reading text from the screen.
                # It slowed down the game loop too much
                # text = read_text_from_region(pil_image, region=(ocr_region[0], ocr_region[1], ocr_region[2], ocr_region[3]), debug=False)
                # print(text)

                # Add the screenshot to the queue for OCR
                image_queue.put((pil_image, ocr_region))

            # Calculate the FPS and update the loop time
            fps, loop_time = calculate_fps(loop_time)

            # Add the current FPS to the accumulator
            fps_accumulator.append(fps)

            # If enough frames have been accumulated, calculate the average FPS
            if len(fps_accumulator) == fps_average_period:
                avg_fps = sum(fps_accumulator) / len(fps_accumulator)
                print(f'Average FPS: {avg_fps:.2f}')
                fps_accumulator = []

            # Check if the 'q' key is pressed to exit the loop
            if cv.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        # Handle any keyboard interrupt (e.g., Ctrl+C) gracefully
        print("Interrupted by user.")
    finally:
        # Signal the OCR thread to stop and wait for it to finish
        ocr_thread_running = False
        ocr_thread.join()

        # Ensure all OpenCV windows are closed when the loop is exited
        cv.destroyAllWindows()
        print('Done.')

# Check if the script is run directly (not imported as a module)
if __name__ == "__main__":
    # Execute the main function
    main()
