import RPi.GPIO as GPIO
from rpi_ws281x import Adafruit_NeoPixel, Color
import torch
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import ast
import queue
import time
from CameraServerClass import CameraServer
from TRSensors import TRSensors
from ServoControllerClass import ServoController
import threading
import multiprocessing as mp

# LED strip configuration constants:
LED_COUNT      = 4      # Number of LED pixels.
LED_PIN        = 18     # GPIO pin connected to the pixels (must support PWM!).
LED_FREQ_HZ    = 800000 # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 5      # DMA channel to use for generating signal (try 5)
LED_BRIGHTNESS = 255    # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0

# Proportional controller constant

KP = 0.3
KD = 1.5
KI = 0.01

CENTER = 2000  # Sensor center value
SPEED = 13
OBJECT_RECOGNITION_WEIGHTS_PATH = "weights.h5"
IMAGENET_LABELS_PATH = "imagenet1000_clsidx_to_labels.txt"

class AlphaBot2(object):
    def __init__(self):
        self.AIN1 = 12
        self.AIN2 = 13
        self.BIN1 = 20
        self.BIN2 = 21
        self.ENA = 6
        self.ENB = 26
        self.PA = 25
        self.PB = 25
        self.integral = 0
        self.last_proportional = 0
        self.maximum = 25
        self.DR = 16
        self.DL = 19
        self.CS = 5
        self.Clock = 25
        self.Address = 24
        self.DataOut = 23
        self.Buzzer = 4
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        motor_pins = [self.AIN1, self.AIN2, self.BIN1, self.BIN2, self.ENA, self.ENB]
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(self.Clock, GPIO.OUT)
        GPIO.setup(self.CS, GPIO.OUT)
        GPIO.setup(self.Address, GPIO.OUT)
        GPIO.setup(self.DataOut, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.Buzzer, GPIO.OUT)
        GPIO.setup(self.DR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.DL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.PWMA = GPIO.PWM(self.ENA, 500)
        self.PWMB = GPIO.PWM(self.ENB, 500)
        self.PWMA.start(0)
        self.PWMB.start(0)
        self.stop()
        # Initialize distance sensors
        self.DR_status = 1
        self.DL_status = 1
        # Initialize additional components
        self.tr_sensor = TRSensors()
        self.servo = ServoController()
        self.servo.center()


        # LED Strip Initialization
        self.led_strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ,
                                            LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.led_strip.begin()
        # Object recognition now runs in the vision process.
        self.object_model = None
        self.imagenet_classes = None

    def setMotor(self, left, right):
        """
        left/right: -100 to +100
        positive = forward
        negative = backward
        """

        # clamp values
        left = max(-100, min(100, left))
        right = max(-100, min(100, right))

        # LEFT MOTOR
        if left >= 0:
            GPIO.output(self.AIN1, GPIO.LOW)
            GPIO.output(self.AIN2, GPIO.HIGH)
            self.PWMA.ChangeDutyCycle(left)
        else:
            GPIO.output(self.AIN1, GPIO.HIGH)
            GPIO.output(self.AIN2, GPIO.LOW)
            self.PWMA.ChangeDutyCycle(-left)

        # RIGHT MOTOR
        if right >= 0:
            GPIO.output(self.BIN1, GPIO.LOW)
            GPIO.output(self.BIN2, GPIO.HIGH)
            self.PWMB.ChangeDutyCycle(right)
        else:
            GPIO.output(self.BIN1, GPIO.HIGH)
            GPIO.output(self.BIN2, GPIO.LOW)
            self.PWMB.ChangeDutyCycle(-right)

    # SAFE STOP
    def stop(self):
        self.PWMA.ChangeDutyCycle(0)
        self.PWMB.ChangeDutyCycle(0)

        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.LOW)

    # compatibility aliases for old code
    def setPWMA(self, duty):
        self.PWMA.ChangeDutyCycle(max(0, min(100, duty)))

    def setPWMB(self, duty):
        self.PWMB.ChangeDutyCycle(max(0, min(100, duty)))

    def load_object_recognition_model(self):
        self.object_model, self.imagenet_classes = load_object_recognition_model()

    def set_led(self, index, r, g, b):
        """Set a single LED's color."""
        if 0 <= index < LED_COUNT:
            self.led_strip.setPixelColor(index, Color(r, g, b))

    def update_leds(self):
        """Update the LED strip to show the current colors."""
        self.led_strip.show()

    def clear_leds(self):
        """Turn off all LEDs."""
        for i in range(LED_COUNT):
            self.led_strip.setPixelColor(i, Color(0, 0, 0))
        self.led_strip.show()

    def set_leds_default(self):
        """Set a default pattern on the LED strip."""
        #self.set_led(0, 255, 0, 0)    # Red
        #self.set_led(1, 0, 255, 0)    # Green
        #self.set_led(2, 0, 0, 255)    # Blue
        #self.set_led(3, 255, 255, 0)  # Yellow
        self.update_leds()
        time.sleep(2)
        self.clear_leds()

    def infrared_obstacle_check(self):
        self.DR_status = GPIO.input(self.DR)
        self.DL_status = GPIO.input(self.DL)
        return self.DL_status == 0 or self.DR_status == 0

    def buzzer_on(self):
        GPIO.output(self.Buzzer, GPIO.HIGH)

    def buzzer_off(self):
        GPIO.output(self.Buzzer, GPIO.LOW)

    def apply_recognition_result(self, result):
        semantic_label = result.get("semantic_label")

        if semantic_label == "shoe":
            self.set_led(0, 255, 0, 0)
        elif semantic_label == "mug":
            self.set_led(1, 255, 255, 0)
        elif semantic_label == "bottle":
            self.set_led(2, 0, 255, 0)

        self.update_leds()

    # Follow Line
    def follow_line(self):
        position, sensors = self.tr_sensor.readLine()
        self.setMotor(SPEED, SPEED)
        proportional = position - CENTER
        derivative = proportional - self.last_proportional
        self.integral += proportional
        self.last_proportional = proportional
        power_difference = (KP * proportional) #+ (KI * self.integral) #+ (KD * derivative)

        ### Line recovery
        # black_count = sum(1 for v in sensors if v < 400)
        # if black_count == 0:
        # decide direction from last known error
        # if self.last_proportional > 0:
            # self.setMotor(SPEED, -SPEED)   # search right
        # else:
            # self.setMotor(-SPEED, SPEED)   # search left
        # self.integral = 0
        # continue

        self.setMotor(SPEED - power_difference, SPEED + power_difference)

def load_object_recognition_model():
    try:
        object_model = models.quantization.mobilenet_v2(
            weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,
            quantize=True
        )
        object_model.eval()

        with open(IMAGENET_LABELS_PATH, "r") as f:
            labels_dict = ast.literal_eval(f.read())
            imagenet_classes = [labels_dict[i] for i in range(len(labels_dict))]

        print("Object recognition model loaded successfully.")
        return object_model, imagenet_classes
    except Exception as e:
        print("Error loading object recognition model:", e)
        return None, None

def map_imagenet_class_to_robot_label(class_id):
    shoe_classes = {502, 514, 630, 770, 774, 788}
    mug_classes = {504, 647, 968}
    bottle_classes = {440, 720, 737, 898, 907}

    if class_id in shoe_classes:
        return "shoe"
    if class_id in mug_classes:
        return "mug"
    if class_id in bottle_classes:
        return "bottle"
    return None

def classify_frame(model, imagenet_classes, frame):
    if model is None or imagenet_classes is None:
        print("Object recognition model not loaded. Cannot recognize object.")
        return None
    if frame is None:
        print("No frame captured for object recognition.")
        return None

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Ensure correct input size
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch)
        probs = output[0].softmax(dim=0)
        top_prob, top_idx = torch.max(probs, dim=0)
        class_id = top_idx.item()
        class_name = imagenet_classes[class_id]
        probability = top_prob.item()
        semantic_label = map_imagenet_class_to_robot_label(class_id)

        print(f"Object Recognition: {probability * 100:.2f}% {class_name}")
        if semantic_label is not None:
            print(f"{semantic_label}!")

        return {
            "class_id": class_id,
            "class_name": class_name,
            "probability": probability,
            "semantic_label": semantic_label,
        }

def enqueue_latest_only(target_queue, item):
    try:
        if target_queue.full():
            target_queue.get_nowait()
        target_queue.put_nowait(item)
    except Exception as e:
        pass

def get_latest_if_available(source_queue):
    try:
        return source_queue.get_nowait()
    except Exception as e:
        return None

 #########################################################################

def camera_process(frame_queue, stop_event, camera_ready):
    camera_server = None
    try:
        camera_server = CameraServer(frame_queue=frame_queue)
        camera_server.start_server()
        camera_ready.set()
        print("Camera server started. Visit http://<your_pi_ip>:5000/ in your browser.")
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        if camera_server is not None:
            camera_server.stop_server()

def vision_process(frame_queue, result_queue, stop_event, vision_ready):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    object_model, imagenet_classes = load_object_recognition_model()
    if object_model is None or imagenet_classes is None:
        return

    vision_ready.set()

    while not stop_event.is_set():
        try:
            frame_msg = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        result = classify_frame(object_model, imagenet_classes, frame_msg.get("frame"))
        if result is None:
            continue

        result["timestamp"] = frame_msg.get("timestamp")
        enqueue_latest_only(result_queue, result)

        time.sleep(0.2)

def beep_pattern(bot, count, stop_event):
    for _ in range(count):
        if stop_event.is_set():
            break
        bot.buzzer_on()
        time.sleep(0.2)
        bot.buzzer_off()
        time.sleep(0.4)

    bot.buzzer_off()


def main_process():
    stop_event = mp.Event()
    camera_ready = mp.Event()
    vision_ready = mp.Event()
    frame_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)

    camera_proc = mp.Process(
        target=camera_process,
        args=(frame_queue, stop_event, camera_ready),
        daemon=True,
    )
    vision_proc = mp.Process(
        target=vision_process,
        args=(frame_queue, result_queue, stop_event, vision_ready),
        daemon=True,
    )

    bot = None
    beep_thread = None

    try:
        camera_proc.start()
        vision_proc.start()

        if not camera_ready.wait(timeout=30):
            print("Camera process failed to start.")
            return
        if not vision_ready.wait(timeout=60):
            print("Vision process failed to start.")
            return

        GPIO.cleanup()
        bot = AlphaBot2()
        bot.set_led(2, 0, 0, 255)    # Blue
        bot.buzzer_on()
        time.sleep(0.1)
        bot.buzzer_off()
        time.sleep(2)
        bot.clear_leds()

        ####### CALIBRATION PHASE
        print("Calibrating... move robot over line")
        # Manual
        #while True:
        #    print(bot.tr_sensor.AnalogRead())
        #    time.sleep(0.1)

        # Automatic
        # for i in range(200):
            # if (i // 50) % 2 == 0:
                # bot.setMotor(10, -10)
            # else:
                # bot.setMotor(-10, 10)

            # bot.tr_sensor.calibrate()
            # time.sleep(0.02)

        # bot.stop()
        # print("Min:", bot.tr_sensor.calibratedMin)
        # print("Max:", bot.tr_sensor.calibratedMax)

        # print("Calibration done")
        # bot.tr_sensor.calibratedMin = [164, 142, 176, 138, 177]
        # bot.tr_sensor.calibratedMax = [971, 973, 975, 970, 978]
        # 183, 206 , 218 , 467 , 464
        #
        bot.tr_sensor.calibratedMin = [210, 193, 218, 184, 247]
        bot.tr_sensor.calibratedMax = [956, 957, 960, 951, 949]

        print("Min:", bot.tr_sensor.calibratedMin)
        print("Max:", bot.tr_sensor.calibratedMax)

        detected_object = 1
        obstacle_was_present = False

        print("Started camera process, vision process, and control loop")
        while not stop_event.is_set():
            obstacle_detected = bot.infrared_obstacle_check()

            if obstacle_detected:
                if not obstacle_was_present:
                    if beep_thread is None or not beep_thread.is_alive():
                        beep_thread = threading.Thread(
                            target=beep_pattern,
                            args=(bot, detected_object, stop_event),
                            daemon=True,
                        )
                        beep_thread.start()
                        if detected_object < 3:
                            detected_object += 1

                    

                obstacle_was_present = True
            else:
                obstacle_was_present = False

            bot.follow_line()

            result = get_latest_if_available(result_queue)
            if result is not None:
                bot.apply_recognition_result(result)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping execution.")
        stop_event.set()
    finally:
        stop_event.set()

        if beep_thread is not None and beep_thread.is_alive():
            beep_thread.join(timeout=1)

        camera_proc.join(timeout=3)
        vision_proc.join(timeout=3)

        if bot is not None:
            bot.buzzer_off()
            bot.stop()
            bot.servo.stop()
        GPIO.cleanup()
        print("All operations stopped. Exiting program.")


if __name__ == '__main__':
    main_process()
