# Chess Position Detection API

A REST API that extracts chess positions from images and returns Lichess editor URLs.

## Features

- Detects chess pieces from board images
- Uses corner coordinates for accurate board rectification
- Generates FEN notation
- Provides Lichess editor URLs
- Deployed on Railway

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "Chess Position Detection API is running",
  "version": "1.0.0"
}
```

### `POST /detect`
Detect chess position from an image.

**Parameters:**
- `image` (file): Chess board image
- `corners` (form field): JSON string with corner coordinates in one of two formats:
- `original_width` (optional, integer): Width of the image where corners were marked
- `original_height` (optional, integer): Height of the image where corners were marked

**Corner Format 1 - Array (simple):**
```json
[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
```
Order: Any 4 corners (algorithm auto-sorts them)

**Corner Format 2 - Dictionary (explicit):**
```json
{
  "topLeft": {"x": 592.6, "y": 1018.2},
  "topRight": {"x": 1065.0, "y": 1148.0},
  "bottomRight": {"x": 792.9, "y": 1508.5},
  "bottomLeft": {"x": 278.0, "y": 1291.0}
}
```

**Image Resizing Feature:**
If you mark corners on an image displayed at one resolution (e.g., 1920x1080 in the browser) but the actual uploaded file has different dimensions (e.g., 3840x2160), provide `original_width=1920` and `original_height=1080`. 

The API will **resize the uploaded image** to match the dimensions where corners were marked. This ensures corners always align correctly regardless of the uploaded image resolution!

**Example Use Cases:**
- Frontend displays images scaled to fit the screen
- User clicks corners on the displayed (scaled) image
- Upload the full-resolution file + provide displayed dimensions
- API resizes to match, corners work perfectly!

**Response example:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "lichess_url": "https://lichess.org/editor/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1",
  "board_matrix": [
    ["Black-Rook", "Black-Knight", ...],
    ...
  ]
}
```

## Local Development

### Prerequisites
- Python 3.10+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/andreea-lipan/chess-ai-python-backend
cd chess-position-api
```

2. Create a virtual environment: (Or let pycharm do it automatically)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file from example:
```bash
cp .env.example .env
```

5. Add your Roboflow API key to `.env`:
```env
ROBOFLOW_API_KEY=your_actual_api_key_here
```

6. Run the API: (Or directly from Pycharm)
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## Deployment to Railway

The application is deployed and the API is available at `https://chess-ai-python-backend-production.up.railway.app/`


## Project Structure

```
chess-position-api/
├── app.py              # Main application
├── tests.py            # Application tests
├── requirements.txt    # Python dependencies (for railway)
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md           
``` 

## How It Works

1. **Image Upload**: Client sends chess board image + corner coordinates
2. **Rectification**: Image is perspective-transformed using corners
3. **Piece Detection**: Roboflow model detects pieces on rectified board
4. **Grid Mapping**: Pieces are mapped to 8x8 chess grid
5. **FEN Generation**: Board state is converted to FEN notation
6. **URL Creation**: Lichess editor URL is generated

## Tests

### Accuracy Metrics

1. **Occupancy Accuracy** - Measures how many squares were correctly identified as occupied vs empty, regardless of which piece is on them
2. **Shape Accuracy** - Measures how many pieces were correctly identified by type (Pawn, Knight, Bishop, etc.), regardless of color
3. **Piece Accuracy** - Measures exact matches including both piece type and color
4. **Full Position Match** - Checks if the entire board position matches perfectly

### Performance Metrics

1. Average processing time per image
2. Min/max processing times

### Error Analysis

1. Total error count across all tests
2. Common confusion pairs (which pieces are frequently misidentified)
3. Per-position errors with square-level details

### Test Data
The test suite uses a predefined set of 12 chessboard images with known ground truth FEN positions, testing various:

1. Board orientations (above or in front)
2. Piece configurations (more or less pieces, on the back ranks or not, etc.)

### Outputs:

1. Console output with test results
2. JSON file with detailed results and metrics (to maintain the history of the tests)
