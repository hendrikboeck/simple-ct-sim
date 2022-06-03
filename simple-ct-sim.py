# -- STL
import asyncio
import time
import sys
import os
import argparse
from multiprocessing import Pool
from typing import Tuple
from functools import partial

# -- LIBRARY
from PIL import Image
import numpy as np

EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1

IMG_SIZE: int = 512  #px
MAX_GRAY_VALUE: float = 255.0
MIN_GRAY_VALUE: float = 0.0

OUTPUT_DIR: str = "./export"


##
# prints an error message to stderr, with colored prefix
#
# @param  *values  list of values that should be printed as error message
def print_error(*values: object) -> None:
  print(f"\033[1;91mERR!\033[0m", *values, file=sys.stderr)


##
# Parses arguments from the command line and returns them as an
# argsparse.Namespace.
#
# @return Read values from command line.
def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Simple CT Simulation")
  parser.add_argument(
      "--runs",
      type=int,
      nargs="+",
      metavar="N",
      help="array of simulations that should be run. Number specifies the"
      " number of pictures taken in that simulation run.",
      required=True
  )
  parser.add_argument(
      "--src",
      type=str,
      help="square graylevel, which CT should be simulated on",
      required=True
  )
  parser.add_argument(
      "--out",
      type=str,
      help="output directory for images",
      default=OUTPUT_DIR
  )

  return parser.parse_args()


async def ct_scan_from_angle(org: Image.Image, angle: float) -> np.ndarray:
  sub = org.rotate(angle)
  sub = np.asarray(sub) / MAX_GRAY_VALUE
  res = np.zeros(org.height)

  for i in range(len(sub)):
    res[i] = np.sum(sub[i])

  return res


async def generate_ct_scan(src: Image.Image, num_scans: int) -> np.ndarray:
  theta = 360.0 / num_scans
  scans = np.zeros((num_scans, IMG_SIZE))
  tasks = []

  for i in range(num_scans):
    tasks.append(asyncio.create_task(ct_scan_from_angle(src, i * theta)))

  for i, task in enumerate(tasks):
    scans[i] = await task

  return scans


async def create_reprojection_array(
    scl: np.ndarray, theta: float
) -> np.ndarray:
  rea = np.empty((IMG_SIZE, IMG_SIZE))

  for i in range(IMG_SIZE):
    rea[i] = scl

  rea = Image.fromarray(rea)
  rea = rea.rotate(theta - 90.0)

  return np.asarray(rea)


async def parallel_reconstruction(scans: np.ndarray) -> Image.Image:
  theta = 360.0 / len(scans)
  scans = scans / np.amax(scans)
  tasks = []

  for i in range(len(scans)):
    tasks.append(
        asyncio.create_task(create_reprojection_array(scans[i], i * theta))
    )

  reimg = np.zeros((IMG_SIZE, IMG_SIZE))
  for t in tasks:
    reimg += await t

  reimg = np.flip(reimg, axis=1)
  reimg = (reimg / np.amax(reimg)) * MAX_GRAY_VALUE
  return Image.fromarray(reimg)


async def simulate_ct_scan(org: Image,
                           num_scans: int) -> Tuple[np.ndarray, Image.Image]:
  scans = await generate_ct_scan(org, num_scans)
  reimg = await parallel_reconstruction(scans)

  return scans, reimg


def create_scan_beam_image(scans: np.ndarray) -> Image.Image:
  # generate scan beam image from raw scan lines array
  scans = (scans / np.amax(scans)) * MAX_GRAY_VALUE
  num_scans = len(scans)
  scbimg = np.zeros((IMG_SIZE, IMG_SIZE))

  for x in range(IMG_SIZE):
    x_i = int(float(x) / IMG_SIZE * num_scans)
    for y in range(IMG_SIZE):
      scbimg[y, x] = scans[x_i, y]

  # convert image data from array to PIL image
  return Image.fromarray(scbimg)


##
# Thread for proccessing a ct scan.
#
# @param  src   source image in PIL format
# @param  num_scans   number of scans taken from source image
def ct_scan_thread(src: Image.Image, num_scans: int) -> None:
  print(f"Started simulation {num_scans}.")
  # get starttime of proccess
  t0 = time.process_time()

  # simulate specific ct scan
  scans, reimg = asyncio.run(simulate_ct_scan(src, num_scans))

  # convert reconstructed image from floating point to int greyscale image
  reimg = reimg.convert("L")
  # save reconstruced image as jpeg
  reimg.save(f"{OUTPUT_DIR}/ct-sim-p{num_scans}.jpg", "JPEG", quality=80)

  scbimg = create_scan_beam_image(scans)
  # convert scan beam from floating point to int greyscale image
  scbimg = scbimg.convert("L")
  # save scan beam image
  scbimg.save(f"{OUTPUT_DIR}/ct-scb-p{num_scans}.jpg", "JPEG", quality=80)

  # scale image to 512x512
  if IMG_SIZE != 512:
    scbimg = scbimg.thumbnail((512, 512), Image.ANTIALIAS)
    reimg = reimg.thumbnail((512, 512), Image.ANTIALIAS)

  # copy scan beam and reconstructed image onto graph image
  compimg = Image.open("import/graph.png").convert("RGB")
  compimg.paste(scbimg, (75, 100))
  compimg.paste(reimg, (662, 100))
  # save composed image as jpeg
  compimg.save(f"{OUTPUT_DIR}/graph-p{num_scans}.jpg", "JPEG", quality=80)

  # measure simulation time
  dt = time.process_time() - t0
  print(f"Completed simulation {num_scans} in {dt:.2f}s.")


##
# Main function.
#
# @param  args  arguments parsed from command line
# @return exit code for program
def main(args: argparse.Namespace) -> int:
  global IMG_SIZE, OUTPUT_DIR

  # create output directory, if it does not exist
  os.makedirs(args.output, exist_ok=True)
  OUTPUT_DIR = args.output

  # check if source files exists
  if not os.path.exists(args.input):
    print_error(f"Path to input file '{args.input}' could not be found.")
    return EXIT_FAILURE
  # import and convert source image to PIL greyvalue format
  org = Image.open(args.input).convert("L")
  # check if image is square
  if org.height != org.width:
    print_error(f"Image has to be square shape, not {org.height}x{org.width}.")
    return EXIT_FAILURE
  # save image size in global variable
  IMG_SIZE = org.height

  # remove duplicates from list
  runs = list(set(args.runs))

  # print program and configuration information
  print("Simple CT Simulation:")
  print(f" * Output Directory: {OUTPUT_DIR}")
  print(f" * Image Path: {args.input}")
  print(f" * Image Size: {IMG_SIZE} (px)")
  print(f" * Runs: {runs}")

  # run simulations simultaniously
  thread_args = [(org, run) for run in runs]
  with Pool(len(runs)) as pool:
    pool.starmap(ct_scan_thread, thread_args)

  # exit main and return success
  return EXIT_SUCCESS


# run main function and exit with returned code.
if __name__ == "__main__":
  sys.exit(main(parse_arguments()))