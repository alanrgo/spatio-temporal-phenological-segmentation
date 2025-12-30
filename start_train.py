import tensorflow as tf
from temporal_segmentation_ecoInfo_v2 import main
from temporal_segmentation import main as serra_cipo_main
import os

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

# main(
#   "/home/alangomes/data/Itirapina/Itirapina/v2/raw/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/aspidosperma/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/aspidosperma/",
#   0.01,
#   0.005,
#   100,
#   50,
#   25,
#   "A.tomentosum",
#   "training"
# )



# main(
#   "/home/alangomes/data/Itirapina/Itirapina/v2/raw/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/all/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/all/",
#   0.01,
#   0.005,
#   100,
#   200000,
#   25,
#   "all",
#   "training"
# )

# main(
#   "/home/alangomes/data/Itirapina/Itirapina/v2/raw/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/all/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/all/",
#   0.01,
#   0.005,
#   100,
#   200000,
#   25,
#   "all",
#   "training"
# )

# main(
#   "/home/alangomes/data/Itirapina/Itirapina/v2/raw/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/all/",
#   "/home/alangomes/data/Itirapina/Itirapina/v2/output/all/model_22000",
#   0.01,
#   0.005,
#   100,
#   200000,
#   25,
#   "all",
#   "testing"
# )

def filter_filenames(filenames):
  """Filters filenames to include 'crop' and exclude 'Cópia de'."""
  filtered_list = [f for f in filenames if 'crop' in f and 'Cópia de' not in f]
  return filtered_list


def has_c(filename):
  return 'c' in filename

def has_mask(filename):
  return 'mask' in filename

PATH = '/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/Imagens_Cedro_ePhenoWS'
file_list = os.listdir(PATH)
masks_filenames = filter(has_mask, file_list)
c_filenames = filter(has_c, file_list)

crop_filenames = filter_filenames(file_list)

## NEEDS ACTIVATION OF COVNNET

# serra_cipo_main(
#   "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/Imagens_Cedro_ePhenoWS/",
#   "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/output/",
#   "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/output/",
#   crop_filenames,
#   0.0003, 
#   0.005, 
#   32, 
#   200000, 
#   25,
#   'training'
# )

serra_cipo_main(
  "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/Imagens_Cedro_ePhenoWS/",
  "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/output/",
  "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/output/model_153000",
  crop_filenames,
  0.0003, 
  0.005, 
  32, 
  200000, 
  25,
  'training'
)

# serra_cipo_main(
#   "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/Imagens_Cedro_ePhenoWS/",
#   "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/output/",
#   "/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/output/model_153000",
#   crop_filenames,
#   0.0003, 
#   0.005, 
#   32, 
#   200000, 
#   25,
#   'testing'
# )
