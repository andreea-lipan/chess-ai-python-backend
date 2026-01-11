# -*- coding: utf-8 -*-
"""Chess Position Detection API"""

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
PIECE_PROJECT = "chessv1-ghvlw"
PIECE_VERSION = 3
RECTIFIED_SIZE = (800, 800)

# Log startup configuration (without exposing the key)
logger.info("="*60)
logger.info("Chess Position Detection API Starting")
logger.info("="*60)
logger.info(f"ROBOFLOW_API_KEY present: {bool(ROBOFLOW_API_KEY)}")
logger.info(f"Corner Project: {CORNER_PROJECT} v{CORNER_VERSION}")
logger.info(f"Piece Project: {PIECE_PROJECT} v{PIECE_VERSION}")
logger.info(f"Rectified Size: {RECTIFIED_SIZE}")
logger.info("="*60)

# Initialize FastAPI
app = FastAPI(
    title="Chess Position Detection API",
    description="Extract chess positions from images and get Lichess URLs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class ChessPositionResponse(BaseModel):
    fen: str
    lichess_url: str
    board_matrix: List[List[str]]

# Global model cache
_corner_model = None
_piece_model = None

def load_models():
    """Load models with caching."""
    global _corner_model, _piece_model
    
    logger.info("Loading Roboflow models...")
    
    if _corner_model is None or _piece_model is None:
        if not ROBOFLOW_API_KEY:
            logger.error("ROBOFLOW_API_KEY environment variable not set!")
            raise ValueError("ROBOFLOW_API_KEY environment variable not set")
        
        try:
            logger.info("Initializing Roboflow client...")
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            
            logger.info(f"Loading corner detection model: {CORNER_PROJECT} v{CORNER_VERSION}")
            _corner_model = rf.workspace().project(CORNER_PROJECT).version(CORNER_VERSION).model
            
            logger.info(f"Loading piece detection model: {PIECE_PROJECT} v{PIECE_VERSION}")
            _piece_model = rf.workspace().project(PIECE_PROJECT).version(PIECE_VERSION).model
            
            logger.info("✓ Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}", exc_info=True)
            raise
    else:
        logger.info("Using cached models")
    
    return _corner_model, _piece_model

def rectify_board(image_path, corners, output_size=RECTIFIED_SIZE):
    """Performs perspective transform."""
    logger.info("Starting board rectification...")
    logger.info(f"Input corners: {corners}")
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image from {image_path}")
            raise ValueError("Failed to read image")
        
        logger.info(f"Image shape: {img.shape}")
        
        # Handle different corner formats
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
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img, M, (w, h))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            cv2.imwrite(tmp.name, warped)
            logger.info(f"✓ Rectified image saved to {tmp.name}")
            return tmp.name
    except Exception as e:
        logger.error(f"Board rectification failed: {str(e)}", exc_info=True)
        raise

def detect_pieces(piece_model, rectified_path):
    """Detects chess pieces."""
    logger.info("Starting piece detection...")
    
    try:
        results = piece_model.predict(rectified_path, confidence=25).json()
        logger.info(f"Piece detection complete: {len(results['predictions'])} pieces found")
        
        # Class mapping
        class_map = {
            0: "Black-Bishop",
            1: "Black-King",
            2: "Black-Knight",
            3: "Black-Pawn",
            4: "Black-Queen",
            5: "Black-Rook",
            6: "White-Bishop",
            7: "White-King",
            8: "White-Knight",
            9: "White-Pawn",
            10: "White-Queen",
            11: "White-Rook",
        }
        
        # Add class names to predictions
        for i, res in enumerate(results["predictions"]):
            class_id = res['class']
            if isinstance(class_id, str) and class_id.isdigit():
                class_id = int(class_id)
            
            if isinstance(class_id, int) and class_id in class_map:
                res['class_name'] = class_map[class_id]
                logger.debug(f"Piece {i+1}: {class_map[class_id]} at ({res['x']:.1f}, {res['y']:.1f})")
            else:
                res['class_name'] = f"Unknown-{class_id}"
                logger.warning(f"Unknown class ID: {class_id}")
        
        logger.info("✓ Piece detection complete")
        return results["predictions"]
    except Exception as e:
        logger.error(f"Piece detection failed: {str(e)}", exc_info=True)
        raise

def map_pieces_to_grid(predictions, board_size=800, grid=8):
    """Convert detections to an 8x8 board matrix."""
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

def label_to_symbol(label):
    """Convert piece labels to FEN symbols."""
    if label == "empty":
        return None
    
    mapping = {
        "White-Pawn": "P", "White-Knight": "N", "White-Bishop": "B",
        "White-Rook": "R", "White-Queen": "Q", "White-King": "K",
        "Black-Pawn": "p", "Black-Knight": "n", "Black-Bishop": "b",
        "Black-Rook": "r", "Black-Queen": "q", "Black-King": "k",
    }
    
    return mapping.get(label, None)

def board_to_fen(board_matrix):
    """Convert board matrix to FEN string."""
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

def fen_to_lichess_url(fen):
    """Convert FEN to Lichess editor URL."""
    return "https://lichess.org/editor/" + fen.replace(" ", "_")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Chess Position Detection API is running",
        "version": "1.0.0"
    }

@app.post("/debug/image")
async def debug_image(image: UploadFile = File(...)):
    """
    Debug endpoint to verify image upload.
    Returns image metadata and allows you to download it back.
    """
    from fastapi.responses import StreamingResponse
    import io
    
    logger.info("="*60)
    logger.info("DEBUG IMAGE UPLOAD")
    logger.info("="*60)
    
    try:
        # Read image
        content = await image.read()
        
        # Log metadata
        metadata = {
            "filename": image.filename,
            "content_type": image.content_type,
            "size_bytes": len(content),
            "size_kb": round(len(content) / 1024, 2),
            "size_mb": round(len(content) / (1024 * 1024), 2)
        }
        
        logger.info(f"Filename: {metadata['filename']}")
        logger.info(f"Content-Type: {metadata['content_type']}")
        logger.info(f"Size: {metadata['size_bytes']} bytes ({metadata['size_kb']} KB)")
        
        # Try to load with OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            img = cv2.imread(tmp_path)
            if img is not None:
                metadata["opencv_loaded"] = True
                metadata["image_shape"] = img.shape
                metadata["width"] = img.shape[1]
                metadata["height"] = img.shape[0]
                metadata["channels"] = img.shape[2] if len(img.shape) > 2 else 1
                metadata["dtype"] = str(img.dtype)
                
                logger.info(f"✓ OpenCV loaded successfully")
                logger.info(f"  Dimensions: {metadata['width']}x{metadata['height']}")
                logger.info(f"  Channels: {metadata['channels']}")
            else:
                metadata["opencv_loaded"] = False
                metadata["error"] = "OpenCV could not decode the image"
                logger.error("✗ OpenCV failed to load image")
        except Exception as e:
            metadata["opencv_loaded"] = False
            metadata["error"] = str(e)
            logger.error(f"✗ Error loading with OpenCV: {e}")
        finally:
            os.unlink(tmp_path)
        
        logger.info("="*60)
        
        # Return both metadata and the image itself
        return {
            "status": "success",
            "metadata": metadata,
            "note": "Image received successfully. Check logs for detailed analysis."
        }
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect", response_model=ChessPositionResponse)
async def detect_chess_position(
    image: UploadFile = File(...),
    corners: str = Form(...)
):
    """
    Detect chess position from image.
    
    Args:
        image: Chess board image file
        corners: JSON string of corner coordinates, e.g. "[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    
    Returns:
        ChessPositionResponse with FEN, Lichess URL, and board matrix
    """
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info("="*60)
    logger.info(f"NEW REQUEST [{request_id}]")
    logger.info("="*60)
    logger.info(f"Image filename: {image.filename}")
    logger.info(f"Image content type: {image.content_type}")
    logger.info(f"Corners received: {corners}")
    
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
        
        # Optional: Log image properties for debugging
        try:
            img_check = cv2.imread(image_path)
            if img_check is not None:
                logger.info(f"Image successfully loaded - Shape: {img_check.shape}, Dtype: {img_check.dtype}")
                logger.info(f"Image dimensions: {img_check.shape[1]}x{img_check.shape[0]} pixels")
                logger.info(f"Color channels: {img_check.shape[2] if len(img_check.shape) > 2 else 1}")
            else:
                logger.error(f"⚠️ OpenCV failed to read the uploaded image at {image_path}")
        except Exception as img_err:
            logger.warning(f"Could not inspect image properties: {img_err}")
        
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
        
        logger.info("="*60)
        logger.info(f"SUCCESS [{request_id}]")
        logger.info(f"FEN: {fen}")
        logger.info(f"URL: {url}")
        logger.info("="*60)
        
        return ChessPositionResponse(
            fen=fen,
            lichess_url=url,
            board_matrix=grid
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("="*60)
        logger.error(f"ERROR [{request_id}]")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error("="*60, exc_info=True)
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

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)