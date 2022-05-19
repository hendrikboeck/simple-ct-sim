import asyncio
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
from typing import Tuple
import threading

IMG_SIZE = 512


async def ct_scan_from_angle(org: PILImage, angle: float) -> np.ndarray:
  sub = org.rotate(angle)
  sub = np.asarray(sub, dtype=float)
  res = np.zeros(IMG_SIZE, dtype=float)

  for i in range(len(sub)):
    res[i] = np.amax(sub[i])

  return res


def parallel_reconstruction(scans: np.ndarray, angle_step: float) -> PILImage:
  normalized_scans = scans / float(len(scans))
  img = Image.new("F", (IMG_SIZE, IMG_SIZE), 0)

  for r in range(len(normalized_scans)):
    # rerotate image every iteration to prevent no rotation if angle is to small
    img = img.rotate(r * angle_step)
    for i in range(IMG_SIZE):
      for j in range(IMG_SIZE):
        img.putpixel(
            (i, j),
            img.getpixel((i, j)) + float(normalized_scans[r, j])
        )
    img = img.rotate(-1.0 * r * angle_step)

  return img


async def simulate_ct_scan(org: Image,
                           num_pictures: int) -> Tuple[np.ndarray, PILImage]:
  angle_step = 360.0 / num_pictures
  scans = np.zeros((num_pictures, IMG_SIZE), dtype=float)
  tasks = []

  for i in range(num_pictures):
    tasks.append(asyncio.create_task(ct_scan_from_angle(org, i * angle_step)))

  for i, task in enumerate(tasks):
    row_scan = await task
    scans[i] = row_scan

  return scans, parallel_reconstruction(scans, angle_step)


def ct_scan_thread(org: PILImage, num_pictures: int) -> None:
  raw, reimg = asyncio.run(simulate_ct_scan(org, num_pictures))

  reimg = reimg.convert("L")
  reimg.save(f"export/ct-sim-p{num_pictures}.jpg", "JPEG")

  scbimg = Image.new("F", (IMG_SIZE, IMG_SIZE), 0)
  for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):

      scbimg.putpixel(
          (i, j),
          scbimg.getpixel((i, j)) +
          float(raw[int((float(i) / IMG_SIZE) * len(raw)), j])
      )
  scbimg = scbimg.convert("L")
  scbimg.save(f"export/ct-scb-p{num_pictures}.jpg", "JPEG")

  reimg = reimg.convert("RGB")
  scbimg = scbimg.convert("RGB")
  compimg = Image.open("import/graph.png").convert("RGB")

  compimg.paste(scbimg, (75, 100))
  compimg.paste(reimg, (662, 100))
  compimg.save(f"export/graph-p{num_pictures}.jpg", "JPEG")


def main() -> None:
  org = Image.open("import/org.png").convert("L")
  threads = []
  #runs = [1, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 128, 256, 360, 720]
  runs = [1, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 128, 256]

  for i in runs:
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