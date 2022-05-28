import asyncio
import time
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
from typing import Tuple
import threading

IMG_SIZE = 512


async def ct_scan_from_angle(org: PILImage, angle: float) -> np.ndarray:
  sub = org.rotate(angle)
  sub = np.asarray(sub, dtype=float) / 255.0
  res = np.zeros(IMG_SIZE, dtype=float)

  for i in range(len(sub)):
    res[i] = np.sum(sub[i])

  return res


async def create_reprojection_array(
    scl: np.ndarray, theta: float
) -> np.ndarray:
  des = np.empty((512, 512))

  for i in range(512):
    des[i] = scl

  des = Image.fromarray(des)
  des = des.rotate(theta - 90.0)

  return np.asarray(des)


async def parallel_reconstruction(
    scans: np.ndarray, angle_step: float
) -> PILImage:
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
    arr = await t
    des += arr

  des = np.flip(des, axis=1)
  des = (des / np.amax(des)) * 255.0
  return Image.fromarray(des)


async def simulate_ct_scan(org: Image,
                           num_pictures: int) -> Tuple[np.ndarray, PILImage]:
  angle_step = 360.0 / num_pictures
  scans = np.zeros((num_pictures, IMG_SIZE), dtype=float)
  tasks = []

  for i in range(num_pictures):
    tasks.append(asyncio.create_task(ct_scan_from_angle(org, i * angle_step)))

  for i, task in enumerate(tasks):
    scans[i] = await task

  des = await parallel_reconstruction(scans, angle_step)
  return scans, des


def ct_scan_thread(org: PILImage, num_pictures: int) -> None:
  raw, reimg = asyncio.run(simulate_ct_scan(org, num_pictures))

  reimg = reimg.convert("L")
  reimg.save(f"export/ct-sim-p{num_pictures}.jpg", "JPEG")

  raw = (raw / np.amax(raw)) * 255.0
  scbimg = np.zeros((512, 512))
  for y in range(IMG_SIZE):
    for x in range(IMG_SIZE):
      scbimg[y, x] += float(raw[int((x / 512.0) * num_pictures), y])
  scbimg = Image.fromarray(scbimg).convert("L")
  scbimg.save(f"export/ct-scb-p{num_pictures}.jpg", "JPEG")

  compimg = Image.open("import/graph.png").convert("RGB")
  compimg.paste(scbimg, (75, 100))
  compimg.paste(reimg, (662, 100))
  compimg.save(f"export/graph-p{num_pictures}.jpg", "JPEG")


def main() -> None:
  org = Image.open("import/org.png").convert("L")
  threads = []
  runs = [1, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 128, 256, 360, 720]
  #runs = [1, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 128, 256]
  #runs = [24]

  t0 = time.process_time()
  for i in runs:
    t = threading.Thread(target=ct_scan_thread, args=(
        org,
        i,
    ))
    t.start()
    threads.append(t)

  for t in threads:
    t.join()
  print(time.process_time() - t0)


if __name__ == "__main__":
  main()