import sys
import os
import colorsys

import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as rletools
import pdb
from PIL import Image
from multiprocessing import Pool
#from io import load_sequences, load_seqmap
from functools import partial
from subprocess import call
import cv2
import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import glob
import os
import pdb


class SegmentedObject:
  def __init__(self, mask, class_id, track_id):
    self.mask = mask
    self.class_id = class_id
    self.track_id = track_id


def load_sequences(path, seqmap):
  objects_per_frame_per_sequence = {}
  for seq in seqmap:
    print("Loading sequence", seq)
    seq_path_folder = os.path.join(path, seq)
    seq_path_txt = os.path.join(path, seq + ".txt")
    print(seq_path_folder)
    if os.path.isdir(seq_path_folder):
      objects_per_frame_per_sequence[seq] = load_images_for_folder(seq_path_folder)
    elif os.path.exists(seq_path_txt):
      objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
    else:
      assert False, "Can't find data in directory " + path

  return objects_per_frame_per_sequence


def load_txt(path):
  objects_per_frame = {}
  track_ids_per_frame = {}  # To check that no frame contains two objects with same id
  combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      fields = line.split(" ")

      frame = int(fields[0])
      if frame not in objects_per_frame:
        objects_per_frame[frame] = []
      if frame not in track_ids_per_frame:
        track_ids_per_frame[frame] = set()
      if int(fields[1]) in track_ids_per_frame[frame]:
        assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
      else:
        track_ids_per_frame[frame].add(int(fields[1]))

      class_id = int(fields[2])
      if not(class_id == 1 or class_id == 2 or class_id == 10):
        assert False, "Unknown object class " + fields[2]

      mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
      if frame not in combined_mask_per_frame:
        combined_mask_per_frame[frame] = mask
      elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
        assert False, "Objects with overlapping masks in frame " + fields[0]
      else:
        combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)
      objects_per_frame[frame].append(SegmentedObject(
        mask,
        class_id,
        int(fields[1])
      ))

  return objects_per_frame


def load_images_for_folder(path):
  files = sorted(glob.glob(os.path.join(path, "*.png")))

  objects_per_frame = {}
  for file in files:
    objects = load_image(file)
    frame = filename_to_frame_nr(os.path.basename(file))
    objects_per_frame[frame] = objects

  return objects_per_frame


def filename_to_frame_nr(filename):
  assert len(filename) == 10, "Expect filenames to have format 000000.png, 000001.png, ..."
  return int(filename.split('.')[0])


def load_image(filename, id_divisor=1000):
  img = np.array(Image.open(filename))
  obj_ids = np.unique(img)
  print(obj_ids)

  objects = []
  mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
  for idx, obj_id in enumerate(obj_ids):
    if obj_id == 0:  # background
      continue
    mask.fill(0)
    pixels_of_elem = np.where(img == obj_id)
    mask[pixels_of_elem] = 1
    objects.append(SegmentedObject(
      rletools.encode(mask),
      obj_id // id_divisor,
      obj_id
    ))

  return objects


def load_seqmap(seqmap_filename):
  print("Loading seqmap...")
  seqmap = []
  max_frames = {}
  with open(seqmap_filename, "r") as fh:
   # pdb.set_trace()
    for i, l in enumerate(fh):
      fields = l.split(" ")
      if fields[0].find("MOTS20-")>-1:
          #fields[0]=fields[0].replace("MOTS20-","00")
          seq = fields[0]#"%04d" % int(fields[0][1:])
      else: 
          seq = "%04d" % int(fields[0][1:])
      seqmap.append(seq)
      max_frames[seq] = int(fields[3])
    print(seqmap,max_frames)
  return seqmap, max_frames


def write_sequences(gt, output_folder):
  os.makedirs(output_folder, exist_ok=True)
  for seq, seq_frames in gt.items():
    write_sequence(seq_frames, os.path.join(output_folder, seq + ".txt"))
  return


def write_sequence(frames, path):
  with open(path, "w") as f:
    for t, objects in frames.items():
      for obj in objects:
        print(t, obj.track_id, obj.class_id, obj.mask["size"][0], obj.mask["size"][1],
              obj.mask["counts"].decode(encoding='UTF-8'), file=f)
# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  N = 30
  brightness = 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6, 10]
  colors = [colors[idx] for idx in perm]
  return colors


# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
  """Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] * (1 - alpha) + alpha * color[c],
                              image[:, :, c])
  return image


def process_sequence(seq_id, tracks_folder, img_folder, output_folder, max_frames, draw_boxes=False,
                     create_video=True):
  print("Processing sequence", seq_id)
  os.makedirs(output_folder + "/" + seq_id, exist_ok=True)
  tracks = load_sequences(tracks_folder, [seq_id])[seq_id]
  max_frames_seq = max_frames[seq_id]
  visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, output_folder, draw_boxes, create_video)


def visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, output_folder, draw_boxes=False, create_video=True):
  colors = generate_colors()
  dpi = 100.0
  frames_with_annotations = [frame for frame in tracks.keys() if len(tracks[frame]) > 0]
 # pdb.set_trace()
  img_sizes = rletools.decode(next(iter(tracks[frames_with_annotations[0]])).mask).shape
  for t in range(max_frames_seq + 1):
    print("Processing frame", t)
    filename_t = img_folder + "/" + seq_id + "/%06d" % t
    if os.path.exists(filename_t + ".png"):
      filename_t = filename_t + ".png"
    elif os.path.exists(filename_t + ".jpg"):
      filename_t = filename_t + ".jpg"
    else:
      print("Image file not found for " + filename_t + ".png/.jpg, continuing...")
      continue
    img = np.array(Image.open(filename_t), dtype="float32") / 255
    fig = plt.figure()
    fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.subplots()
    ax.set_axis_off()

    if t in tracks:
      for obj in tracks[t]:
        color = colors[obj.track_id % len(colors)]
        if obj.class_id == 1:
          category_name = "Car"
        elif obj.class_id == 2:
          category_name = "Pedestrian"
        else:
          category_name = "Ignore"
          color = (0.7, 0.7, 0.7)
        if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
          x, y, w, h = rletools.toBbox(obj.mask)
          if draw_boxes:
            import matplotlib.patches as patches
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                     edgecolor=color, facecolor='none', alpha=1.0)
            ax.add_patch(rect)
          category_name += ":" + str(obj.track_id)
          ax.annotate(category_name, (x + 0.5 * w, y + 0.5 * h), color=color, weight='bold',
                      fontsize=7, ha='center', va='center', alpha=1.0)
        binary_mask = rletools.decode(obj.mask)
        apply_mask(img, binary_mask, tuple([x for x in color]))
        img=img[:,:,]

    ax.imshow(img)
    #cv2.imwrite(output_folder + "/" + seq_id + "/%06d" % t + ".png",img[:,:,[2,1,0]])
    fig.savefig(output_folder + "/" + seq_id + "/%06d" % t + ".png")

    plt.close(fig)
  if create_video:
    os.chdir(output_folder + "/" + seq_id)
    call(["ffmpeg", "-framerate", "10", "-y", "-i", "%06d.jpg", "-c:v", "libx264", "-profile:v", "high", "-crf", "20",
          "-pix_fmt", "yuv420p", "-vf", "pad=\'width=ceil(iw/2)*2:height=ceil(ih/2)*2\'", "output.mp4"])


def main():
  if len(sys.argv) != 5:
    print("Usage: python visualize_mots.py tracks_folder(gt or tracker results) img_folder output_folder seqmap")
    sys.exit(1)

  tracks_folder = sys.argv[1]
  img_folder = sys.argv[2]
  output_folder = sys.argv[3]
  seqmap_filename = sys.argv[4]

  seqmap, max_frames = load_seqmap(seqmap_filename)
  process_sequence_part = partial(process_sequence, max_frames=max_frames,
                                  tracks_folder=tracks_folder, img_folder=img_folder, output_folder=output_folder)

  with Pool(10) as pool:
    pool.map(process_sequence_part, seqmap)
  #for seq in seqmap:
  #  process_sequence_part(seq)


if __name__ == "__main__":
  main()
