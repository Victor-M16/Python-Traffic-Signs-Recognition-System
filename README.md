# Python Traffic Signs Recognition System

This project is a **Traffic Sign Classification System** built using Python, TensorFlow/Keras, and Tkinter. It allows users to upload an image of a traffic sign, and the system will classify the sign using a pre-trained deep learning model.

---

## Features

- **User-friendly GUI**: Built with Tkinter for an intuitive user experience.
- **Deep learning-powered classification**: Uses a trained `traffic_classifier.h5` model for accurate predictions.
- **Predefined traffic sign classes**: Supports the classification of 43 different traffic signs based on a standardized dataset.
- **Model Improvement Capability**: Allows training on new images to enhance model accuracy and robustness.

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+ 
- Required Python libraries (install via `requirements.txt`)

### Installation

1. **Clone the Repository**  
   Clone the project to your local machine:
   ```bash
   git clone https://github.com/Victor-M16/Python-Traffic-Signs-Recognition-System.git
   cd Python-Traffic-Signs-Recognition-System
   ```

2. **Install Dependencies**  
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   Launch the GUI application:
   ```bash
   python gui.py
   ```

---

## Usage

### Traffic Sign Classification

1. **Start the Application**  
   Run the application and wait for the GUI to appear.

2. **Upload an Image**  
   - Click the **"Upload an image"** button.
   - Use the file explorer to select an image of a traffic sign.

3. **Classify the Image**  
   - Click the **"Classify Image"** button to analyze the uploaded image.
   - The predicted traffic sign will be displayed at the bottom of the GUI.

### Model Improvement with New Images

This project supports **incremental training** to improve the model using new images. Follow these steps:

1. **Prepare Your Data**  
   - Organize new traffic sign images into folders named according to their class IDs (e.g., `0`, `1`, `2`, etc.).
   - Place these folders in a directory named `train`.

2. **Update the Model**  
   - Run the training script provided in the repository to include new images in the training process:
     ```bash
     python train.py
     ```
   - The script will:
     - Load existing data and the new images.
     - Preprocess the data by resizing images to 30x30 pixels and normalizing them.
     - Retrain the model with both old and new data.

3. **Replace the Old Model**  
   - The updated model will be saved as `my_model.h5`.
   - Replace the existing `traffic_classifier.h5` in the `gui.py` code with `my_model.h5` to use the improved version.

---

## Classes

The system can recognize the following traffic signs:

1. Speed limit (20km/h)  
2. Speed limit (30km/h)  
3. Speed limit (50km/h)  
...  
*(Complete list available in the source code.)*

---

## File Structure

- **`gui.py`**: Main application script for the GUI.
- **`train.py`**: Script for training the model with new or additional images.
- **`traffic_classifier.h5`**: Pre-trained model for classifying traffic signs.
- **`requirements.txt`**: Python dependencies for the project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## Acknowledgments

- The pre-trained model is based on a standardized traffic signs dataset.
- Tkinter was used to design the GUI for easy interaction.

Enjoy using the Traffic Sign Classification System! 🚦
