YOLO_UNET

Deep learning based segmentation proposed in 
"Narotamo, H., Sanches, J. M., & Silveira, M. (2019, July). Segmentation of Cell Nuclei in Fluorescence Microscopy Images Using Deep Learning. In Iberian Conference on Pattern Recognition and Image Analysis (pp. 53-64). Springer, Cham.".

Code adapted from: https://github.com/thtrieu/darkflow

## How to run

runfile(r"C:\Users\hemaxi\Desktop\darkflow-master\setup.py",args="build_ext --inplace")
(where C:\Users\hemaxi\Desktop\darkflow-master is the directory where the setup.py file is saved)

run flow --model cfg/tiny-yoloc1_1024.cfg --imgdir IMG_DIR --load -1 --json
(where IMG_DIR is the directory where the test image is saved)

## How to cite
```bibtex
@inproceedings{narotamo2019segmentation,
  title={Segmentation of cell nuclei in fluorescence microscopy images using deep learning},
  author={Narotamo, Hemaxi and Sanches, J Miguel and Silveira, Margarida},
  booktitle={Iberian Conference on Pattern Recognition and Image Analysis},
  pages={53--64},
  year={2019},
  organization={Springer}
}
```
