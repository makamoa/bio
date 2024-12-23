# Real-Time Simultaneous Refractive Index and Thickness Mapping of Sub-Cellular Biology at the Diffraction Limit

Authors:  
- Arturo Burguete-Lopez
- Maksim Makarenko
- Marcella Bonifazi
- Barbara Nicoly Menezes de Oliveira
- Fedor Getman
- Yi Tian
- Valerio Mazzone
- Ning Li
- Alessandro Giammona
- Carlo Liberale
- Andrea Fratalocchi

Published in: Communications Biology, Volume 7, Article number: 154 (2024)  
[Link to Article](https://www.nature.com/articles/s42003-024-05839-w)

---

## Abstract

![image](https://github.com/user-attachments/assets/6240840f-0cda-4a2d-848f-d91a77ffcf04)

This repository provides code for real-time mapping of cellular refractive index (RI) and thickness at diffraction-limited resolutions. This innovative machine-learning-based technique can map the RI and thickness of biological specimens using a single image from a conventional color camera. By utilizing a nanostructured membrane that stretches a biological analyte over its surface, this technology achieves high RI sensitivity (10⁻⁴) and sub-nanometer thickness resolution.

## Key Features

- **Real-time RI and thickness mapping**: Achieves simultaneous mapping at diffraction-limited spatial resolutions.
- **High sensitivity and resolution**: 10⁻⁴ RI sensitivity and sub-nanometer thickness resolution.
- **Single-image acquisition**: RI and thickness are obtained from a single image using a color camera, without pre-existing sample knowledge.
- **Biomedical application**: Enables sub-cellular segmentation and 3D reconstruction of cellular regions, demonstrated on HCT-116 colorectal cancer cells.
- **Machine-learning-based**: Employs ML models to process complex reflection spectra from the nanostructured membrane.

## Repository Contents

- `aux/`: Auxiliary files for data correspondence and analysis.
- `mats/`: Contains material and origin data for model configurations.
- `src/`: Source code, including scripts for data processing and clustering. Key files include `ImageClustering` and other utilities for data handling.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `CameraCalibration.ipynb`: Notebook for calibrating camera settings for image analysis.
- `FlatOptimization.ipynb`: Notebook for flat optimization analysis of the imaging process.
- `ImageClustering.ipynb`: Notebook tutorial for clustering biological images based on refractive index and thickness characteristics.
- `ImageOptimization.ipynb`: Notebook for optimizing image data for refractive index and thickness mapping.
- `README.md`: This readme file, providing an overview of the project, usage instructions, and setup details.

## Getting Started

### Prerequisites

- Python 3.11
- Key libraries: `numpy`, `scipy`, `opencv-python`, `torch`, `matplotlib`
- For visualization and analysis, `jupyter` is recommended

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/realtime-RI-thickness-mapping.git
   cd realtime-RI-thickness-mapping
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Preparation**: Place your image files in the `data/` directory. Follow the data format provided in the sample dataset.
2. **Training the Model**: Run `scripts/train.py` to train the model on the provided dataset.
   ```bash
   python scripts/train.py --config config/train_config.yaml
   ```
3. **Inference**: Use `scripts/inference.py` to generate RI and thickness maps from new images.
   ```bash
   python scripts/inference.py --input data/new_image.jpg --output results/
   ```
4. **Visualization**: Use notebooks in `notebooks/` to visualize the output RI and thickness maps.

### Configuration

Model and training parameters can be modified in the YAML configuration files located in the `config/` directory. For custom configurations, create a new `.yaml` file based on the provided templates.

## Results

This approach achieves accurate RI and thickness mapping for real-time label-free imaging of biological specimens. The models included allow for reproducible analysis of cellular components and 3D reconstruction of cellular regions, demonstrated on colorectal cancer cells with sub-cellular segmentation.

## Citation

If you use this repository, please cite the following article:

Burguete-Lopez, A., Makarenko, M., Bonifazi, M., Menezes de Oliveira, B.N., Getman, F., Tian, Y., Mazzone, V., Li, N., Giammona, A., Liberale, C., & Fratalocchi, A. (2024). Real-time simultaneous refractive index and thickness mapping of sub-cellular biology at the diffraction limit. *Communications Biology, 7*, Article 154. [https://doi.org/10.1038/s42003-024-0154-0](https://doi.org/10.1038/s42003-024-0154-0)

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations, please contact [Maksim Makarenko](mailto:makarenko@kaust.edu.sa) or any of the co-authors listed in the publication.

---

Thank you for using our code repository for real-time RI and thickness mapping!
