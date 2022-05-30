import asyncio
import time
from PIL import Image
import numpy as np
from typing import Tuple
import threading
from typing import NoReturn
import sys
import argparse
import os

IMG_SIZE: int = 512  #px
MAX_GRAY_VALUE: float = 255.0
MIN_GRAY_VALUE: float = 0.0

OUTPUT_DIR: str = "./export"


def error_exit(*values: object) -> NoReturn:
  sys.exit("ERROR: ", *values)


async def ct_scan_from_angle(org: Image.Image, angle: float) -> np.ndarray:
  sub = org.rotate(angle)
  sub = np.asarray(sub) / MAX_GRAY_VALUE
  res = np.zeros(org.height)

  for i in range(len(sub)):
    res[i] = np.sum(sub[i])

  return res


async def create_reprojection_array(
    scl: np.ndarray, theta: float
) -> np.ndarray:
  des = np.empty((IMG_SIZE, IMG_SIZE))

  for i in range(IMG_SIZE):
    des[i] = scl

  des = Image.fromarray(des)
  des = des.rotate(theta - 90.0)

  return np.asarray(des)


async def parallel_reconstruction(
    scans: np.ndarray, angle_step: float
) -> Image.Image:
  scans = scans / np.amax(scans)
  tasks = []

  for i in range(len(scans)):
    tasks.append(
        asyncio.create_task(
            create_reprojection_array(scans[i], i * angle_step)
        )
    )

  des = np.zeros((IMG_SIZE, IMG_SIZE))
  for t in tasks:
    des += await t

  des = np.flip(des, axis=1)
  des = (des / np.amax(des)) * MAX_GRAY_VALUE
  return Image.fromarray(des)


async def simulate_ct_scan(org: Image,
                           num_pictures: int) -> Tuple[np.ndarray, Image.Image]:
  angle_step = 360.0 / num_pictures
  scans = np.zeros((num_pictures, IMG_SIZE))
  tasks = []

  for i in range(num_pictures):
    tasks.append(asyncio.create_task(ct_scan_from_angle(org, i * angle_step)))

  for i, task in enumerate(tasks):
    scans[i] = await task

  des = await parallel_reconstruction(scans, angle_step)
  return scans, des


def ct_scan_thread(org: Image.Image, num_pictures: int) -> None:
  t0 = time.process_time()

  raw, reimg = asyncio.run(simulate_ct_scan(org, num_pictures))

  reimg = reimg.convert("L")
  reimg.save(f"{OUTPUT_DIR}/ct-sim-p{num_pictures}.jpg", "JPEG")

  raw = (raw / np.amax(raw)) * MAX_GRAY_VALUE
  scbimg = np.zeros((IMG_SIZE, IMG_SIZE))
  for y in range(IMG_SIZE):
    for x in range(IMG_SIZE):
      scbimg[y, x] += float(raw[int(float(x) / IMG_SIZE * num_pictures), y])
  scbimg = Image.fromarray(scbimg).convert("L")
  scbimg.save(f"{OUTPUT_DIR}/ct-scb-p{num_pictures}.jpg", "JPEG")

  if IMG_SIZE != 512:
    scbimg = scbimg.thumbnail((512, 512), Image.ANTIALIAS)
    reimg = reimg.thumbnail((512, 512), Image.ANTIALIAS)

  compimg = Image.open("import/graph.png").convert("RGB")
  compimg.paste(scbimg, (75, 100))
  compimg.paste(reimg, (662, 100))
  compimg.save(f"{OUTPUT_DIR}/graph-p{num_pictures}.jpg", "JPEG")

  dt = time.process_time() - t0
  print(f"Thread #{num_pictures} completed! took {dt}s")


def main() -> None:
  global IMG_SIZE, OUTPUT_DIR

  parser = argparse.ArgumentParser(description="simple ct simulation")
  parser.add_argument(
      "--runs",
      type=int,
      nargs="+",
      metavar="N",
      help=
      "array of simulations that should be run. Number specifies the number of"
      " pictures taken in that simulation run.",
      required=True
  )
  parser.add_argument(
      "--input",
      type=str,
      help="square graylevel, which CT should be simulated on",
      required=True
  )
  parser.add_argument(
      "--output",
      type=str,
      help="output directory for images",
      default=OUTPUT_DIR
  )
  args = parser.parse_args()

  if not os.path.exists(args.output):
    os.makedirs(args.output, exist_ok=True)
  OUTPUT_DIR = args.output

  if not os.path.exists(args.input):
    error_exit("path to input file does not exist.")
  org = Image.open(args.input).convert("L")
  if org.height != org.width:
    error_exit("image has to be square.")
  IMG_SIZE = org.height

  threads = []

  for i in args.runs:
    t = threading.Thread(target=ct_scan_thread, args=(
        org,
        i,
    ))
    t.start()
    threads.append(t)

  for t in threads:
    t.join()


if __name__ == "__main__":
  main()