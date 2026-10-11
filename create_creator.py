# create_creator.py
import math
import os
import random

random.seed(42)

N = 200  # grid points per side
SIZE = 100.0  # world size in meters
SP = SIZE / (N - 1)
CX = CY = SIZE / 2  # crater center = world origin (0, 0)

R_BIG = 42.0  # main crater radius
DEPTH = 7.0  # floor depth below the surrounding plain
RIM = 2.0  # rim crest height


def place(min_r, max_r):
    """Random point inside the crater, between min_r and max_r from the center."""
    r = math.sqrt(random.uniform(min_r**2, max_r**2))
    a = random.uniform(0, 2 * math.pi)
    return CX + r * math.cos(a), CY + r * math.sin(a)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def main_crater(r):
    s = r / R_BIG
    if s < 1.0:
        wall = smoothstep((s - 0.45) / 0.55)
        return -DEPTH * (1 - wall) + RIM * s**6  # flat floor, steep wall, sharp rim
    return RIM * s**-3  # ejecta blanket fades outward


# Everything below sits inside the big crater. The center (r < 8) stays clear.

# small craters: (cx, cy, radius, depth, rim_height)
small_craters = []
for _ in range(14):
    cx, cy = place(12, 0.85 * R_BIG)
    R = random.uniform(1.5, 5.0)
    small_craters.append(
        (cx, cy, R, R * random.uniform(0.3, 0.4), R * random.uniform(0.05, 0.09))
    )

# hills: (cx, cy, radius, height)
hills = []
for _ in range(5):
    cx, cy = place(12, 0.7 * R_BIG)
    hills.append((cx, cy, random.uniform(4, 8), random.uniform(1.0, 2.5)))

# sharp pits: (cx, cy, radius, depth)
pits = []
for _ in range(10):
    cx, cy = place(8, 0.8 * R_BIG)
    pits.append((cx, cy, random.uniform(1.5, 2.5), random.uniform(2, 3.5)))

# cliff mesas: (cx, cy, angle, half_width, half_length, drop)
cliffs = []
for _ in range(3):
    cx, cy = place(14, 0.6 * R_BIG)
    cliffs.append(
        (
            cx,
            cy,
            random.uniform(0, math.pi),
            random.uniform(3, 5),
            random.uniform(5, 8),
            random.uniform(1.5, 3),
        )
    )


def height(x, y):
    h = main_crater(math.hypot(x - CX, y - CY))
    for cx, cy, R, d, rim in small_craters:
        s = math.hypot(x - cx, y - cy) / R
        if s < 1.0:
            h += -d * (1 - s * s) + rim * s**4
        else:
            h += rim * s**-3
    for cx, cy, R, H in hills:
        h += H * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (R * R))
    for cx, cy, R, D in pits:
        r = math.hypot(x - cx, y - cy)
        h -= D * math.exp(-((r / R) ** 4))
    for cx, cy, a, W, L, drop in cliffs:
        dx, dy = x - cx, y - cy
        across = dx * math.cos(a) + dy * math.sin(a)
        along = -dx * math.sin(a) + dy * math.cos(a)
        h += (
            drop
            * 0.5
            * (1 + math.tanh((W - abs(across)) / 0.4))
            * 0.5
            * (1 + math.tanh((L - abs(along)) / 0.4))
        )
    h += 0.02 * math.sin(x * 1.3) * math.cos(y * 1.7)  # small roughness
    return h


heights = [height(i * SP, j * SP) for j in range(N) for i in range(N)]
SPAWN_X = 0.0
SPAWN_Y = -25.0
spawn_z = height(SPAWN_X + SIZE / 2, SPAWN_Y + SIZE / 2) + 1.4

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD_DIR = os.path.join(HERE, "worlds")
os.makedirs(WORLD_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(WORLD_DIR, "crater_world.wbt")

with open(OUTPUT_PATH, "w") as f:
    f.write(f"""#VRML_SIM R2025a utf8
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/adept/pioneer3/protos/Pioneer3at.proto"

WorldInfo {{
  basicTimeStep 16
}}
Viewpoint {{
  orientation -0.3 0.3 0.9 1.6
  position -20 -20 30
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
}}
Solid {{
  translation {-SIZE/2} {-SIZE/2} 0
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.55 0.53 0.5
        roughness 1
        metalness 0
      }}
      geometry DEF CRATERS ElevationGrid {{
        xDimension {N}
        xSpacing {SP}
        yDimension {N}
        ySpacing {SP}
        height [
          {" ".join(f"{v:.4f}" for v in heights)}
        ]
      }}
    }}
  ]
  boundingObject USE CRATERS
}}
Pioneer3at {{
  # Interior spawn: yaw only, with zero roll and pitch so the wheels face down.
  translation {SPAWN_X:.1f} {SPAWN_Y:.1f} {spawn_z:.2f}
  # Turn 180 degrees from the original south-facing heading.
  rotation 0 0 1 1.5708
  controller "my_controller"
  extensionSlot [
    Camera {{
      translation 0.25 0 0.3
      name "camera"
      width 640
      height 480
      fieldOfView 1.2
    }}
    SpotLight {{
      location 0.28 0 0.32
      direction 1 0 -0.15
      intensity 8
      beamWidth 0.8
      cutOffAngle 1.0
      radius 40
    }}
    Lidar {{
      translation 0 0 0.45
      name "lidar"
      horizontalResolution 256
      fieldOfView 3.14159
      numberOfLayers 16
      verticalFieldOfView 0.5
      minRange 0.3
      maxRange 30
    }}
    InertialUnit {{
      name "imu"
    }}
    Accelerometer {{
      name "accel"
    }}
    Gyro {{
      name "gyro"
    }}
    GPS {{
      name "gps"
    }}
    Emitter {{
      name "telemetry"
      channel 1
    }}
  ]
}}
Robot {{
  supervisor TRUE
  controller "dashboard"
  children [
    Receiver {{
      name "telemetry"
      channel 1
      samplingPeriod 100
    }}
  ]
}}
""")
print(f"Wrote {OUTPUT_PATH}")
