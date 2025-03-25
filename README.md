# Virtual Drag and Drop

A computer vision application that allows you to manipulate virtual shapes using hand gestures.

![Virtual Drag and Drop Demo](https://github.com/wangqiqi/interesting_assets/raw/main/images/virtual_drag.png)

## Features

- Multiple shape support (Rectangle, Circle, Triangle)
- Intuitive gesture controls:
  - Drag and drop using index and middle fingers
  - Scale shapes using thumb and index finger
  - Rotate shapes while dragging
  - Reset all shapes by making a fist
- Real-time visual feedback:
  - Shape trails and visual effects
  - Mode status display
  - FPS counter
  - Control instructions
- Real-time hand gesture recognition

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd virtual-drag
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python virtual_drag_drop.py
```

### Controls

- **Drag and Drop**: Use index and middle fingers
  - Pinch to grab a shape
  - Move to drag
  - Spread fingers to release
  - Rotate fingers while dragging to rotate the shape

- **Scale**: Use thumb and index finger
  - Pinch near a shape to start scaling
  - Change distance between fingers to scale
  - Spread fingers to stop scaling

- **Reset**: Make a fist to reset all shapes to their initial positions and sizes

- **Exit**: Press 'ESC' to exit the application

## Notes

- Ensure good lighting conditions for better hand detection
- Keep your hand within the camera frame
- Maintain appropriate distance from camera (about 20-50cm)
- The application supports:
  - Multiple shape types
  - Simultaneous rotation and dragging
  - Smooth scaling with limits
  - Visual feedback for active modes

## Contact

If you have any questions or suggestions, feel free to contact me:

- WeChat: znzatop

![WeChat](https://github.com/wangqiqi/interesting_assets/raw/main/images/wechat.jpg)

## More Projects

更多有趣的项目请见：https://github.com/wangqiqi/interesting_assets.git 