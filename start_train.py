import tensorflow as tf
from temporal_segmentation_ecoInfo_v2 import main

tf.keras.backend.clear_session()
# main(
#   "/home/alangomes/data/Itirapina/Itirapina/v1/raw/",
#   "/home/alangomes/data/Itirapina/Itirapina/v1/output/aspidosperma/",
#   "/home/alangomes/data/Itirapina/Itirapina/v1/output/aspidosperma/",
#   0.01,
#   0.005,
#   100,
#   50,
#   25,
#   "aspidosperma",
#   "training"
# )

main(
  "/home/alangomes/data/Itirapina/Itirapina/v2/raw/",
  "/home/alangomes/data/Itirapina/Itirapina/v2/output/aspidosperma/",
  "/home/alangomes/data/Itirapina/Itirapina/v2/output/aspidosperma/",
  0.01,
  0.005,
  100,
  50,
  25,
  "A.tomentosum",
  "training"
)
