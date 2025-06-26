import carla
import cv2
import random
import time
import sys
import math
import joblib
import numpy as np
import os

lower_green = np.array([40, 40, 40])
upper_green = np.array([90, 255, 255])
lower_blue = np.array([85, 10, 150])
upper_blue = np.array([115, 150, 255])

x = 0.3 # scale factor for steering values
pred_to_steer_value = {
    -1.0: -1*x, 
     0.0: 0*x,
     1.0: 1*x
}
# ------------------------ CONFIGURATION --------------------------------------
HOST            = "localhost"
PORT            = 2000
TIMEOUT_S       = 5.0
THROTTLE        = 0.5
DEFAULT_STEER   = 0.0
PRINT_EVERY_N   = 30
MODEL_PATH      = "GroupB_Updated_SVM.joblib"   
IMAGE_SIZE      = (64, 64)
# -----------------------------------------------------------------------------
def predict_steering(img):
    if not hasattr(predict_steering, "_model"):
        model_path = MODEL_PATH
        if not os.path.isfile(model_path):
            predict_steering._model = None
        else:
            predict_steering._model = joblib.load(model_path)
    img_np = np.frombuffer(img.raw_data, dtype=np.uint8)
    img_np = img_np.reshape((img.height, img.width, 4))  
    img_bgr = img_np[:, :, :3]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    combined_mask = cv2.bitwise_or(mask_green, mask_blue)
    height, width = combined_mask.shape
    bottom_half = combined_mask[int(height * 0.5):, :]
    resized = cv2.resize(bottom_half, IMAGE_SIZE)
    features = resized.flatten().reshape(1, -1)
    model = predict_steering._model
    pred_clipped = 0.0
    if model is not None:
        try:
            pred = float(model.predict(features)[0])
        except Exception as e:
            pred = 0.0
            print(f"[ERR] SVM predict failed: {e}")
        pred_clipped = pred_to_steer_value.get(pred, 0.0)
        print(f"SVM steering prediction: {pred_clipped:.3f}")
    return pred_clipped
# -----------------------------------------------------------------
def parent_of(actor):
    if hasattr(actor, "get_parent"):
        return actor.get_parent()
    return getattr(actor, "parent", None)

def ang_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def pick_center_camera(world, vehicle):
    v_yaw = vehicle.get_transform().rotation.yaw
    best = None
    for s in world.get_actors().filter("sensor.camera.rgb"):
        p = parent_of(s)
        if p and p.id == vehicle.id:
            delta = abs(ang_diff_deg(s.get_transform().rotation.yaw, v_yaw))
            if best is None or delta < best[0]:
                best = (delta, s)
    return best[1] if best else None
# -----------------------------------------------------------------------------
def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    world  = client.get_world()

    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        print("No vehicles found. Start a scene first.")
        return
    vehicle = vehicles[0]
    print("Controlling vehicle id=%d type=%s" % (vehicle.id, vehicle.type_id))
    vehicle.set_autopilot(False)

    camera = pick_center_camera(world, vehicle)
    if camera is None:
        print("No center RGB camera attached to the vehicle.")
        return
    print("Using camera id=%d for live feed" % camera.id)

    state = {"frames": 0, "first_ts": None, "latest_img": None}
    prev_steer = [DEFAULT_STEER]  # Use a list to allow modification inside loop

    def cam_cb(img):
        state["latest_img"] = img
        state["frames"] += 1
        if state["frames"] % PRINT_EVERY_N == 0:
            if state["first_ts"] is None:
                state["first_ts"] = img.timestamp
            elapsed = img.timestamp - state["first_ts"]
            fps = state["frames"] / elapsed if elapsed else 0.0
            print("camera frames: %d   %.1f FPS" % (state["frames"], fps))

    camera.listen(cam_cb)

    try:
        while True:
            img = state["latest_img"]
            if img is not None:
                steer_raw = float(max(-1.0, min(1.0, predict_steering(img))))
                # Smooth steering for high speed
                alpha = 0.3
                steer = alpha * steer_raw + (1 - alpha) * prev_steer[0]
                prev_steer[0] = steer
            else:
                steer = DEFAULT_STEER  
            vehicle.apply_control(carla.VehicleControl(throttle=THROTTLE, steer=steer))
            time.sleep(0.005)  # Faster control loop for higher speed

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        camera.stop()
        vehicle.apply_control(carla.VehicleControl(brake=1.0))

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        sys.stderr.write("[ERROR] " + str(err) + "\n"
                         "Is the CARLA server running on this host/port?\n")