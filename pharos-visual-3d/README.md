# Pharos 3D

A real-time 3D visualization system for tracking and displaying devices, humans, and buildings in interactive 3D space. Built with React, Three.js, and modern web technologies.

## Features

- **3D Visualization**: Interactive 3D scene with orbital camera controls
- **Real-time Tracking**: Visualize devices (drones), humans, and buildings with position and velocity data
- **Time-based Playback**: Timeline controls for playing back historical movement data
- **Data Import**: Drag-and-drop JSON file support for loading tracking data
- **Interactive Controls**: Configuration panel for display settings and visualization options

## Getting Started

### Prerequisites

- Node.js (v22 or higher)
- npm or pnpm package manager

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
```

2. Install dependencies:
```bash
pnpm install # or just use npm
```

3. Start the development server:
```bash
npm run dev
```

Open `http://localhost:5173` and drag the `public/vis-demo.json` file to get started.

## Usage

### Quick Start with Demo Data

1. Start the development server (see [Getting Started](#getting-started))
2. Open your browser and navigate to `http://localhost:5173`
3. Download the demo data file: [vis-demo.json](public/vis-demo.json)
4. Drag and drop the `vis-demo.json` file directly onto the application window
5. The 3D scene will load with sample drone and human tracking data
6. Use the timeline controls at the bottom to play/pause and navigate through the simulation

### Data Format

The application accepts JSON files with the following structure:

```json
{
  "devices": [
    {
      "uid": "vehicle/10001",
      "position": [24.0, 48.0, 26.0],
      "velocity": [0.0, 0.0, 0.0],
      "ts": 1742393414649,
      "include_area": [22.0, 46.0, 24.0, 26.0, 50.0, 28.0],
      "target_pos": [24.0, 48.0, 26.0]
    }
  ],
  "humans": [
    {
      "hid": "human/10001",
      "position": [24.0, 1.0, 26.0],
      "velocity": [1.0, 0.0, 0.0],
      "ts": 1742393414649
    }
  ],
  "buildings": [
    {
      "id": "building/10001",
      "bbox": [22.0, 46.0, 24.0, 26.0, 50.0, 28.0]
    }
  ]
}
```

### Controls

- **File Upload**: Drag and drop JSON files directly onto the application window to load new data
- **Timeline**: Use the bottom timeline controls to play/pause and navigate through time
- **Camera**: Click and drag to orbit around the scene, scroll to zoom
- **Settings**: Click the gear icon to access display configuration options
- **Help**: Click the question mark icon for detailed usage instructions

### Keyboard Shortcuts

- `Space`: Play/Pause timeline
- `←/→`: Step backward/forward in time

## Data Structure

### Device Data
- `uid`: Unique device identifier
- `position`: 3D coordinates [x, y, z] where y is height
- `velocity`: Velocity vector [vx, vy, vz]
- `ts`: Timestamp in milliseconds
- `include_area`: Safety zone boundaries [minX, minY, minZ, maxX, maxY, maxZ]
- `target_pos`: Target position [x, y, z]

### Human Data
- `hid`: Unique human identifier
- `position`: 3D coordinates [x, y, z]
- `velocity`: Velocity vector [vx, vy, vz]
- `ts`: Timestamp in milliseconds

### Building Data
- `id`: Unique building identifier
- `bbox`: Bounding box [minX, minY, minZ, maxX, maxY, maxZ]

## Development

### Project Structure

```
src/
├── components/        # React components
│   ├── device-model.tsx
│   ├── human-model.tsx
│   ├── building-model.tsx
│   ├── time-controller.tsx
│   └── config-panel.tsx
├── pages/            # Page components
│   ├── home.tsx
│   └── help.tsx
├── atoms/            # Jotai state atoms
├── hooks/            # Custom React hooks
├── models/           # 3D model components
├── types/            # TypeScript type definitions
└── utils/            # Utility functions
```
