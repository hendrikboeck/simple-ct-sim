import asyncio
from PIL import Image
import numpy as np

IMG_SIZE = 512


async def ct_scan_from_angle(org: Image, angle: float) -> np.ndarray:
  sub = org.rotate(angle)
  sub = np.asarray(sub, dtype=float)
  res = np.zeros(IMG_SIZE, dtype=float)

  for i in range(len(sub)):
    res[i] = np.amax(sub[i])

  return res


def parallel_reconstruction(scans: np.ndarray, angle_step: float) -> Image:
  normalized_scans = scans / float(len(scans))
  img = Image.new("F", (IMG_SIZE, IMG_SIZE), 0)

  for r in range(len(normalized_scans)):
    # rerotate image every iteration to prevent no rotation if angle is to small
    img = img.rotate(r * angle_step)
    for i in range(IMG_SIZE):
      for j in range(IMG_SIZE):
        img.putpixel(
            (i, j), int(img.getpixel((i, j)) + float(normalized_scans[r, j]))
        )
    img = img.rotate(-1.0 * r * angle_step)

  return img


async def simulate_ct_scan(org: Image, num_pictures: int) -> Image:
  angle_step = 360.0 / num_pictures
  scans = np.zeros((num_pictures, IMG_SIZE), dtype=float)
  tasks = []

  for i in range(num_pictures):
    tasks.append(asyncio.create_task(ct_scan_from_angle(org, i * angle_step)))

  for i, task in enumerate(tasks):
    row_scan = await task
    scans[i] = row_scan

  return parallel_reconstruction(scans, angle_step)


async def main() -> None:
  org = Image.open("import/org.png").convert("L")
  tasks = []
  runs = [1, 3, 4, 8, 12, 16, 24, 32, 50, 64, 128]

  for i in runs:
    tasks.append(asyncio.create_task(simulate_ct_scan(org, i)))

  for i, task in enumerate(tasks):
    img = await task
    img = img.convert("L")
    img.save(f"export/ct-sim-p{runs[i]}.jpg", "JPEG")


if __name__ == "__main__":
  asyncio.run(main())