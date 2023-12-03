import cv2 as cv
import numpy as np
from time import time
import pyautogui
from PIL import ImageGrab


def main():

    loop_time = time()
    while True:
        # Capture a screenshot using pyautogui
        screenshot = ImageGrab.grab()
        screenshot = np.array(screenshot)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

        # Display the screenshot in a window named 'Half-Life'
        cv.imshow('game_window', screenshot)

        print('FPS {}'.format(1 / (time() - loop_time)))
        loop_time = time()

        # Press 'q' to exit the loop
        if cv.waitKey(1) == ord('q'):
            break

    # Clean up and close any open windows
    cv.destroyAllWindows()
    print('Done.')

if __name__ == "__main__":
    main()

