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

# Configuration from environment variables
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
CORNER_PROJECT = "chessboard-corner-detection"
CORNER_VERSION = 1
PIECE_PROJECT = "chessv1-ghvlw"
PIECE_VERSION = 3
RECTIFIED_SIZE = (800, 800)

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
    
    if _corner_model is None or _piece_model is None:
        if not ROBOFLOW_API_KEY:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set")
        
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        _corner_model = rf.workspace().project(CORNER_PROJECT).version(CORNER_VERSION).model
        _piece_model = rf.workspace().project(PIECE_PROJECT).version(PIECE_VERSION).model
    
    return _corner_model, _piece_model

def rectify_board(image_path, corners, output_size=RECTIFIED_SIZE):
    """Performs perspective transform."""
    img = cv2.imread(image_path)
    
    # Extract corner coordinates
    pts = np.array(corners, dtype=np.float32)
    
    # Sort corners: TL, TR, BR, BL using sum and diff method
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    
    pts_src = np.float32([tl, tr, br, bl])
    
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
        return tmp.name

def detect_pieces(piece_model, rectified_path):
    """Detects chess pieces."""
    results = piece_model.predict(rectified_path, confidence=25).json()
    
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
    for res in results["predictions"]:
        class_id = res['class']
        if isinstance(class_id, str) and class_id.isdigit():
            class_id = int(class_id)
        
        if isinstance(class_id, int) and class_id in class_map:
            res['class_name'] = class_map[class_id]
        else:
            res['class_name'] = f"Unknown-{class_id}"
    
    return results["predictions"]

def map_pieces_to_grid(predictions, board_size=800, grid=8):
    """Convert detections to an 8x8 board matrix."""
    cell = board_size / grid
    board = [["empty" for _ in range(grid)] for _ in range(grid)]
    
    for p in predictions:
        px, py = p["x"], p["y"]
        col = int(px // cell)
        row = int(py // cell)
        
        piece_name = p.get('class_name', p.get('class', 'Unknown'))
        
        if 0 <= row < 8 and 0 <= col < 8:
            board[row][col] = piece_name
    
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
    board = chess.Board(None)
    
    for r in range(8):
        for c in range(8):
            piece = board_matrix[r][c]
            symbol = label_to_symbol(piece)
            if symbol:
                sq = chess.square(c, 7 - r)
                board.set_piece_at(sq, chess.Piece.from_symbol(symbol))
    
    return board.fen()

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
    temp_files = []
    
    try:
        # Parse corners
        try:
            corner_coords = json.loads(corners)
            if len(corner_coords) != 4:
                raise ValueError("Must provide exactly 4 corner coordinates")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for corners")
        
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            content = await image.read()
            tmp.write(content)
            image_path = tmp.name
            temp_files.append(image_path)
        
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
        
        return ChessPositionResponse(
            fen=fen,
            lichess_url=url,
            board_matrix=grid
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)