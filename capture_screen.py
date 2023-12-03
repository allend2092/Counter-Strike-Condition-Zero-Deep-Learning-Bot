# Import necessary libraries
import cv2 as cv
import numpy as np
from time import time
import pyautogui
from PIL import ImageGrab

def capture_screen():
    """
    Captures the entire screen.

    Returns:
        np.ndarray: The captured screen image in BGR format.
    """
    try:
        # Capture the screen using ImageGrab from PIL
        screenshot = ImageGrab.grab()

        # Convert the screenshot to a NumPy array
        # This is necessary for OpenCV to process the image
        screenshot = np.array(screenshot)

        # Convert the color format from RGB (used by PIL) to BGR (used by OpenCV)
        return cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

    except Exception as e:
        # Print any error that occurs during the screen capture process
        print(f"Error capturing screen: {e}")
        return None

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
    # Initialize the time for FPS calculation
    loop_time = time()

    # Initialize a list to store FPS values for averaging
    fps_accumulator = []

    # Define the number of frames to average for FPS calculation
    fps_average_period = 10

    try:
        while True:
            # Capture the screen
            screenshot = capture_screen()

            # If the screenshot is successfully captured
            if screenshot is not None:
                # Display the screenshot in a window named 'Half-Life'
                cv.imshow('Half-Life', screenshot)

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
        # Ensure all OpenCV windows are closed when the loop is exited
        cv.destroyAllWindows()
        print('Done.')

# Check if the script is run directly (not imported as a module)
if __name__ == "__main__":
    # Execute the main function
    main()
