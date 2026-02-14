# === Chess Position Detection API ===

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from roboflow import Roboflow
import cv2
import numpy as np
import chess
import os
import tempfile
from typing import List
import json
from pydantic import BaseModel
import logging
import sys
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Configuration from environment variables
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
CORNER_PROJECT = "chessboard-corner-detection"
CORNER_VERSION = 1
PIECE_PROJECT = "chess-piece-recognition-zjki1" # AI Alg mine
# PIECE_PROJECT = "chessv1-ghvlw" # AI Alg 1
# PIECE_PROJECT = "chess-piece-vrw2r" # AI Alg 2
PIECE_VERSION = 6 #alg mine
# PIECE_VERSION = 3 #alg 1
# PIECE_VERSION = 7
RECTIFIED_SIZE = (800, 800)


# Log startup configuration
logger.info("=" * 60)
logger.info("Chess Position Detection API Starting")
logger.info("=" * 60)
logger.info(f"ROBOFLOW_API_KEY present: {bool(ROBOFLOW_API_KEY)}")
logger.info(f"Corner Project: {CORNER_PROJECT} v{CORNER_VERSION}")
logger.info(f"Piece Project: {PIECE_PROJECT} v{PIECE_VERSION}")
logger.info(f"Rectified Size: {RECTIFIED_SIZE}")
logger.info("=" * 60)


# Initialize FastAPI
app = FastAPI(
    title="Chess Position Detection API",
    description="Extract chess positions from images and get Lichess URLs",
    version="1.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in the future
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# This is the response the API sends
# The FEN config of the board
# With the corresponding lichess url
class ChessPositionResponse(BaseModel):
    fen: str
    lichess_url: str
    board_matrix: List[List[str]]


# Global model cache
# Note: for now the corner model is disabled, we are giving the corners manually to the API
# We did not find a model accurate and consistent enough to detect the corners of the chess board
_corner_model = None
_piece_model = None



'''
    Load models with caching.
'''
def load_models():
    global _corner_model, _piece_model

    logger.info("Loading Roboflow models...")

    if _corner_model is None or _piece_model is None:
        if not ROBOFLOW_API_KEY:
            logger.error("ROBOFLOW_API_KEY environment variable not set!")
            raise ValueError("ROBOFLOW_API_KEY environment variable not set")

        try:
            logger.info("Initializing Roboflow client...")
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)

            # logger.info(f"Loading corner detection model: {CORNER_PROJECT} v{CORNER_VERSION}")
            # _corner_model = rf.workspace().project(CORNER_PROJECT).version(CORNER_VERSION).model

            logger.info(f"Loading piece detection model: {PIECE_PROJECT} v{PIECE_VERSION}")
            _piece_model = rf.workspace().project(PIECE_PROJECT).version(PIECE_VERSION).model

            logger.info("✓ Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}", exc_info=True)
            raise
    else:
        logger.info("Using cached models")

    return _corner_model, _piece_model


'''
    Performs a board transformation. Crops the image based on the chess board corners.
    Then tilts the image in such a way that the every square from the chess board is the same size.
    This is done to make it easy to detect on what square a piece is located.
'''
def rectify_board(image_path, corners, output_size=RECTIFIED_SIZE):
    logger.info("Starting board rectification...")
    logger.info(f"Input corners: {corners}")

    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image from {image_path}")
            raise ValueError("Failed to read image")

        logger.info(f"Image shape: {img.shape}")

        # Handle different corner formats (we can send them in two ways, this was mostly for testing)
        if isinstance(corners, dict):
            # Dictionary format: {topLeft: {x, y}, topRight: {x, y}, ...}
            logger.info("Detected dictionary corner format")
            pts = np.array([
                [corners['topLeft']['x'], corners['topLeft']['y']],
                [corners['topRight']['x'], corners['topRight']['y']],
                [corners['bottomRight']['x'], corners['bottomRight']['y']],
                [corners['bottomLeft']['x'], corners['bottomLeft']['y']]
            ], dtype=np.float32)

        elif isinstance(corners, list):
            # Array format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            logger.info("Detected array corner format")
            pts = np.array(corners, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported corner format: {type(corners)}")

        # Sort corners: TL, TR, BR, BL using sum and diff method
        # To make sure they are in the correct order
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]

        pts_src = np.float32([tl, tr, br, bl])
        logger.info(f"Ordered corners - TL: {tl}, TR: {tr}, BR: {br}, BL: {bl}")

        # Destination rectangle
        w, h = output_size
        pts_dst = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # Compute transform matrix and warp
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img, matrix, (w, h))

        # For local deploy, save the image in a folder to verify its accuracy in cropping.
        # debug_path = os.path.join(
        #     "/Users/andreea/UNI/Software Engineering/Sem 3/ITSG/chess-ai-python-backend/photos",
        #     f"rectified_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        # )
        #
        # cv2.imwrite(debug_path, warped)
        # logger.info(f"✓ Rectified image saved to {debug_path}")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            cv2.imwrite(tmp.name, warped)
            logger.info(f"✓ Rectified image saved to {tmp.name}")
            return tmp.name

    except Exception as e:
        logger.error(f"Board rectification failed: {str(e)}", exc_info=True)
        raise



'''
    Uses an AI model to detect the chess pieces.
'''
def detect_pieces(piece_model, rectified_path):
    logger.info("Starting piece detection...")

    try:
        results = piece_model.predict(rectified_path, confidence=30).json()
        logger.info(f"Piece detection complete: {len(results['predictions'])} pieces found")

        # Class mapping for AI alg mine
        # class_map = {
        #     0: "black-bishop",
        #     1: "black-king",
        #     2: "black-horse",
        #     3: "black-pawn",
        #     4: "black-queen",
        #     5: "black-rook",
        #     6: "white-bishop",
        #     7: "white-king",
        #     8: "white-horse",
        #     9: "white-pawn",
        #     10: "white-queen",
        #     11: "white-rook",
        # }

        # Class mapping for AI alg 1
        # class_map = {
        #     0: "Black-Bishop",
        #     1: "Black-King",
        #     2: "Black-Knight",
        #     3: "Black-Pawn",
        #     4: "Black-Queen",
        #     5: "Black-Rook",
        #     6: "White-Bishop",
        #     7: "White-King",
        #     8: "White-Knight",
        #     9: "White-Pawn",
        #     10: "White-Queen",
        #     11: "White-Rook",
        # }

        # Class mapping for AI alg 2
        # class_map = {
        #     "Black-bishop": "Black-Bishop",
        #     "Black-king": "Black-King",
        #     "Black-horse": "Black-Knight",
        #     "Black-pawn": "Black-Pawn",
        #     "Black-queen": "Black-Queen",
        #     "Black-rook": "Black-Rook",
        #     "White-bishop": "White-Bishop",
        #     "White-king": "White-King",
        #     "White-horse": "White-Knight",
        #     "White-pawn": "White-Pawn",
        #     "White-queen": "White-Queen",
        #     "White-rook": "White-Rook",
        # }
        class_map = {
            "black-bishop": "black-bishop",
            "black-king": "black-king",
            "black-horse": "black-horse",
            "black-pawn": "black-pawn",
            "black-queen": "black-queen",
            "black-rook": "black-rook",
            "white-bishop": "white-bishop",
            "white-king": "white-king",
            "white-horse": "white-horse",
            "white-pawn": "white-pawn",
            "white-queen": "white-queen",
            "white-rook": "white-rook",
        }

        # Add class names to predictions
        for i, res in enumerate(results["predictions"]):
            class_id = res['class']
            # Code req for AI alg 1 mapping bc it's not strings
            # if isinstance(class_id, str) and class_id.isdigit():
            #     class_id = int(class_id)
            #
            # if isinstance(class_id, int) and class_id in class_map:
            #     res['class_name'] = class_map[class_id]
            #     logger.debug(f"Piece {i + 1}: {class_map[class_id]} at ({res['x']:.1f}, {res['y']:.1f})")
            # else:
            #     res['class_name'] = f"Unknown-{class_id}"
            #     logger.warning(f"Unknown class ID: {class_id}")
            res['class_name'] = class_map[class_id]

        logger.info("✓ Piece detection complete")
        return results["predictions"]
    except Exception as e:
        logger.error(f"Piece detection failed: {str(e)}", exc_info=True)
        raise



'''
    Convert detections to an 8x8 board matrix.
    Figure out on what square is piece on the board.
'''
def map_pieces_to_grid(predictions, board_size=800, grid=8):

    logger.info("Mapping pieces to grid...")
    cell = board_size / grid
    board = [["empty" for _ in range(grid)] for _ in range(grid)]

    mapped_count = 0
    out_of_bounds_count = 0

    for p in predictions:
        px, py = p["x"], p["y"]
        col = int(px // cell)
        row = int(py // cell)

        piece_name = p.get('class_name', p.get('class', 'Unknown'))

        if 0 <= row < 8 and 0 <= col < 8:
            board[row][col] = piece_name
            mapped_count += 1
        else:
            logger.warning(f"Piece {piece_name} at ({px:.1f}, {py:.1f}) -> grid[{row}][{col}] OUT OF BOUNDS")
            out_of_bounds_count += 1

    logger.info(f"✓ Mapped {mapped_count} pieces to grid, {out_of_bounds_count} out of bounds")
    return board



'''
    Convert piece labels to FEN symbols. This is needed for the lichess link.
'''
def label_to_symbol(label):
    if label == "empty":
        return None

    # mapping = {
    #     "White-Pawn": "P", "White-Knight": "N", "White-Bishop": "B",
    #     "White-Rook": "R", "White-Queen": "Q", "White-King": "K",
    #     "Black-Pawn": "p", "Black-Knight": "n", "Black-Bishop": "b",
    #     "Black-Rook": "r", "Black-Queen": "q", "Black-King": "k",
    # }
    mapping = {
        "white-pawn": "P", "white-knight": "N", "white-bishop": "B",
        "white-rook": "R", "white-queen": "Q", "white-king": "K",
        "black-pawn": "p", "black-knight": "n", "black-bishop": "b",
        "black-rook": "r", "black-queen": "q", "black-king": "k",
    }

    return mapping.get(label, None)



'''
    Convert board matrix to FEN string.
'''
def board_to_fen(board_matrix):
    logger.info("Converting board to FEN...")
    board = chess.Board(None)

    piece_count = 0
    for r in range(8):
        for c in range(8):
            piece = board_matrix[r][c]
            symbol = label_to_symbol(piece)
            if symbol:
                sq = chess.square(c, 7 - r)
                board.set_piece_at(sq, chess.Piece.from_symbol(symbol))
                piece_count += 1

    fen = board.fen()
    logger.info(f"✓ FEN generated with {piece_count} pieces: {fen}")
    return fen



'''
    Convert FEN to Lichess URL.
'''
def fen_to_lichess_url(fen):
    return "https://lichess.org/editor/" + fen.replace(" ", "_")



# ============================= API ENDPOINTS =============================


'''
    Base test endpoint.
'''
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Chess Position Detection API is running",
        "version": "1.0.0"
    }



'''
    Detect chess position from image.

    Args:
        image: Chess board image file
        corners: JSON string of corner coordinates, e.g. "[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
        original_width: Optional - Original image width if coordinates are from a different resolution
        original_height: Optional - Original image height if coordinates are from a different resolution

    Returns:
        ChessPositionResponse with FEN, Lichess URL, and board matrix
'''
@app.post("/detect", response_model=ChessPositionResponse)
async def detect_chess_position(
        image: UploadFile = File(...),
        corners: str = Form(...),
        original_width: int = Form(None),
        original_height: int = Form(None)
):

    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info("=" * 60)
    logger.info(f"NEW REQUEST [{request_id}]")
    logger.info("=" * 60)
    logger.info(f"Image filename: {image.filename}")
    logger.info(f"Image content type: {image.content_type}")
    logger.info(f"Corners received: {corners}")
    logger.info(
        f"Original dimensions provided: {original_width}x{original_height}" if original_width else "No original dimensions provided")

    temp_files = []

    try:
        # Parse corners
        try:
            logger.info("Parsing corner coordinates...")
            corner_coords = json.loads(corners)

            # Validate corner format
            if isinstance(corner_coords, dict):
                # Dictionary format
                required_keys = ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']
                if not all(key in corner_coords for key in required_keys):
                    raise ValueError(f"Dictionary format must include: {required_keys}")
                for key in required_keys:
                    if not ('x' in corner_coords[key] and 'y' in corner_coords[key]):
                        raise ValueError(f"{key} must have 'x' and 'y' properties")
                logger.info(f"✓ Parsed corners in dictionary format")

            elif isinstance(corner_coords, list):
                # Array format
                if len(corner_coords) != 4:
                    raise ValueError("Array format must provide exactly 4 corner coordinates")
                logger.info(f"✓ Parsed {len(corner_coords)} corners in array format")
            else:
                raise ValueError("Corners must be either a dictionary or an array")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format for corners: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON format for corners")


        # Save uploaded image to temporary file
        logger.info("Saving uploaded image...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            content = await image.read()
            logger.info(f"Image size: {len(content)} bytes")
            tmp.write(content)
            image_path = tmp.name
            temp_files.append(image_path)
            logger.info(f"✓ Image saved to {image_path}")


        # Resize the image to be the same as the size it was sent at
        # Log image properties for debugging
        try:
            img_check = cv2.imread(image_path)
            if img_check is not None:
                actual_height, actual_width = img_check.shape[:2]
                logger.info(f"Image successfully loaded - Shape: {img_check.shape}, Dtype: {img_check.dtype}")
                logger.info(f"Actual image dimensions: {actual_width}x{actual_height} pixels")
                logger.info(f"Color channels: {img_check.shape[2] if len(img_check.shape) > 2 else 1}")

                # Resize image if original dimensions are provided and different
                logger.info(f"Original image dimensions: {original_width}x{original_height} pixels")
                logger.info(f"Actual image dimensions: {actual_width}x{actual_height} pixels")
                if original_width and original_height:
                    if original_width != actual_width or original_height != actual_height:
                        logger.info(
                            f"Resizing image from {actual_width}x{actual_height} to {original_width}x{original_height}")
                        logger.info("This ensures corner coordinates match the image dimensions")

                        # Resize the image to match the original dimensions
                        resized_img = cv2.resize(img_check, (original_width, original_height),
                                                 interpolation=cv2.INTER_LANCZOS4)

                        # Save the resized image back
                        cv2.imwrite(image_path, resized_img)

                        debug_path = os.path.join(
                            "/Users/andreea/UNI/Software Engineering/Sem 3/ITSG/chess-ai-python-backend/photos",
                            f"resized_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                        )

                        cv2.imwrite(debug_path, resized_img)
                        logger.info(f"✓ Resized image saved to {debug_path}")

                        logger.info(f"✓ Image resized successfully to {original_width}x{original_height}")
                        logger.info(f"Corners will now match the resized image dimensions")
                    else:
                        logger.info("Original dimensions match actual dimensions - no resizing needed")
                else:
                    logger.info("No original dimensions provided - using image as-is")
                    logger.warning("⚠️ If corners don't match, provide original_width and original_height!")
            else:
                logger.error(f"⚠️ OpenCV failed to read the uploaded image at {image_path}")
        except Exception as img_err:
            logger.warning(f"Could not inspect/resize image: {img_err}")



        # ============= MAIN ALG STEPS =============


        # Load models
        corner_model, piece_model = load_models()

        # Rectify board
        rectified_path = rectify_board(image_path, corner_coords)
        temp_files.append(rectified_path)

        # Detect pieces
        piece_preds = detect_pieces(piece_model, rectified_path)

        # Map to 8x8 grid
        grid = map_pieces_to_grid(piece_preds)

        # Convert to FEN
        fen = board_to_fen(grid)

        # Generate Lichess URL
        url = fen_to_lichess_url(fen)

        logger.info("=" * 60)
        logger.info(f"SUCCESS [{request_id}]")
        logger.info(f"FEN: {fen}")
        logger.info(f"URL: {url}")
        logger.info("=" * 60)

        return ChessPositionResponse(
            fen=fen,
            lichess_url=url,
            board_matrix=grid
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"ERROR [{request_id}]")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error("=" * 60, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Deleted {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {temp_file}: {str(e)}")


# for running locally uncomment this and use your devices local IP addr
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.214", port=8000)
