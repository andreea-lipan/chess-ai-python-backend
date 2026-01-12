"""
Chess Position Detection API - Test Suite

Tests piece accuracy, full position match, and performance metrics.
Requires a test dataset with ground truth FEN positions.
"""

import requests
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

# Configuration
API_URL = "http://192.168.0.214:8000"  # Change to Railway URL for production
TEST_DATA_DIR = Path("test_images")
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)


class ChessTestCase:
    """Represents a single test case with ground truth"""

    def __init__(self, image_path: str, corners: List, expected_fen: str,
                 original_width: int = None, original_height: int = None):
        self.image_path = image_path
        self.corners = corners
        self.expected_fen = expected_fen
        self.original_width = original_width
        self.original_height = original_height


# Ground truth test dataset
TEST_CASES = [
    ChessTestCase(
        image_path="test_images/position1.jpeg",
        corners=[[422, 891], [2451, 874], [2468, 3022], [400, 3000]],
        expected_fen="3k4/6r1/3p4/8/8/2Q1P3/3P4/4K3_w_-_-_0_1"
    ),
    ChessTestCase(
        image_path="test_images/position2.jpeg",
        corners=[[1016, 180], [2932, 214], [3258, 2447], [638, 2400]],
        expected_fen="3k4/6r1/3p4/8/8/2Q1P3/3P4/4K3_w_-_-_0_1"
    ),
    ChessTestCase(
        image_path="test_images/position3.jpeg",
        corners=[[366, 137], [1581, 145], [1629, 1356], [349, 1356]],
        expected_fen="8/ppp2pp1/2n4p/3pp3/P3P1p1/2NP1N2/1PP2PPP/8 w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position4.jpeg",
        corners=[[484, 173], [1607, 197], [1875, 1341], [231, 1339]],
        expected_fen="8/ppp2pp1/2n4p/3pp3/P3P1p1/2NP1N2/1PP2PPP/8 w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position5.jpeg",
        corners=[[454, 158], [1588, 180], [1562, 1350], [415, 1326]],
        expected_fen="r2qkbnr/ppp2pp1/2n4p/3pp3/P3P1p1/2NP1N2/1PP2PPP/R1BQKB1R w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position6.jpeg",
        corners=[[486, 111], [1558, 85], [1796, 1260], [272, 1245]],
        expected_fen="r2qkbnr/ppp2pp1/2n4p/3pp3/P3P1p1/2NP1N2/1PP2PPP/R1BQKB1R w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position7.jpeg",
        corners=[[415, 177], [1536, 197], [1579, 1290], [375, 1311]],
        expected_fen="8/8/rbnqkbnr/pppppppp/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position8.jpeg",
        corners=[[460, 186], [1541, 203], [1725, 1236], [297, 1230]],
        expected_fen="8/8/rbnqkbnr/pppppppp/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position9.jpeg",
        corners=[[469, 188], [1515, 218], [1521, 1302], [415, 1290]],
        expected_fen="rbnqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position10.jpeg",
        corners=[[456, 135], [1466, 141], [1684, 1232], [210, 1232]],
        expected_fen="rbnqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position11.jpeg",
        corners=[[501, 177], [1521, 192], [1644, 1298], [381, 1292]],
        expected_fen="8/8/8/2bpBnR1/8/8/2PPP3/8 w - - 0 1"
    ),
    ChessTestCase(
        image_path="test_images/position12.jpeg",
        corners=[[282, 517], [1180, 534], [1365, 1414], [91, 1411]],
        expected_fen="8/8/8/2bpBnR1/8/8/2PPP3/8 w - - 0 1"
    )
]


class ChessAPITester:
    """Test suite for Chess Position Detection API"""

    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.results = []

    def call_api(self, test_case: ChessTestCase) -> Tuple[Dict, float]:
        """Call the API and measure response time"""
        start_time = time.time()

        with open(test_case.image_path, 'rb') as f:
            files = {'image': f}

            data = {'corners': json.dumps(test_case.corners)}

            if test_case.original_width:
                data['original_width'] = test_case.original_width
            if test_case.original_height:
                data['original_height'] = test_case.original_height

            try:
                response = requests.post(
                    f"{self.api_url}/detect",
                    files=files,
                    data=data,
                    timeout=30
                )
                processing_time = time.time() - start_time

                if response.status_code == 200:
                    return response.json(), processing_time
                else:
                    return {'error': f"HTTP {response.status_code}", 'detail': response.text}, processing_time

            except Exception as e:
                processing_time = time.time() - start_time
                return {'error': str(e)}, processing_time

    def fen_to_board_matrix(self, fen: str) -> List[List[str]]:
        """Convert FEN string to 8x8 board matrix"""
        piece_placement = fen.split()[0]

        board = []
        for row in piece_placement.split('/'):
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend(['empty'] * int(char))
                else:
                    piece_map = {
                        'P': 'White-Pawn', 'N': 'White-Knight', 'B': 'White-Bishop',
                        'R': 'White-Rook', 'Q': 'White-Queen', 'K': 'White-King',
                        'p': 'Black-Pawn', 'n': 'Black-Knight', 'b': 'Black-Bishop',
                        'r': 'Black-Rook', 'q': 'Black-Queen', 'k': 'Black-King',
                    }
                    board_row.append(piece_map.get(char, 'unknown'))
            board.append(board_row)

        return board

    def calculate_occupancy_accuracy(self, detected_board: List[List[str]], expected_fen: str) -> Dict:
        """Calculate accuracy of occupied vs empty squares (regardless of piece type)"""
        expected_board = self.fen_to_board_matrix(expected_fen)

        total_squares = 64
        correct_occupancy = 0
        wrong_occupancy = []

        for row in range(8):
            for col in range(8):
                detected = detected_board[row][col]
                expected = expected_board[row][col]

                detected_occupied = detected != 'empty'
                expected_occupied = expected != 'empty'

                if detected_occupied == expected_occupied:
                    correct_occupancy += 1
                else:
                    wrong_occupancy.append({
                        'position': f"{chr(97 + col)}{8 - row}",
                        'detected': 'occupied' if detected_occupied else 'empty',
                        'expected': 'occupied' if expected_occupied else 'empty'
                    })

        accuracy = (correct_occupancy / total_squares) * 100

        return {
            'total_squares': total_squares,
            'correct': correct_occupancy,
            'incorrect': len(wrong_occupancy),
            'accuracy': accuracy,
            'errors': wrong_occupancy
        }

    def calculate_shape_accuracy(self, detected_board: List[List[str]], expected_fen: str) -> Dict:
        """Calculate accuracy of piece shapes (regardless of color)"""
        expected_board = self.fen_to_board_matrix(expected_fen)

        total_squares = 64
        correct_shapes = 0
        wrong_shapes = []

        def get_piece_shape(piece: str) -> str:
            """Extract piece shape without color"""
            if piece == 'empty':
                return 'empty'
            # Extract shape from 'Color-Shape' format
            parts = piece.split('-')
            return parts[1] if len(parts) == 2 else piece

        for row in range(8):
            for col in range(8):
                detected = detected_board[row][col]
                expected = expected_board[row][col]

                detected_shape = get_piece_shape(detected)
                expected_shape = get_piece_shape(expected)

                if detected_shape == expected_shape:
                    correct_shapes += 1
                else:
                    wrong_shapes.append({
                        'position': f"{chr(97 + col)}{8 - row}",
                        'detected': detected_shape,
                        'expected': expected_shape
                    })

        accuracy = (correct_shapes / total_squares) * 100

        return {
            'total_squares': total_squares,
            'correct': correct_shapes,
            'incorrect': len(wrong_shapes),
            'accuracy': accuracy,
            'errors': wrong_shapes
        }

    def calculate_piece_accuracy(self, detected_board: List[List[str]], expected_fen: str) -> Dict:
        """Calculate piece-by-piece accuracy"""
        expected_board = self.fen_to_board_matrix(expected_fen)

        total_squares = 64
        correct_pieces = 0
        wrong_pieces = []

        for row in range(8):
            for col in range(8):
                detected = detected_board[row][col]
                expected = expected_board[row][col]

                if detected == expected:
                    correct_pieces += 1
                else:
                    wrong_pieces.append({
                        'position': f"{chr(97 + col)}{8 - row}",
                        'detected': detected,
                        'expected': expected
                    })

        accuracy = (correct_pieces / total_squares) * 100

        return {
            'total_squares': total_squares,
            'correct': correct_pieces,
            'incorrect': len(wrong_pieces),
            'accuracy': accuracy,
            'errors': wrong_pieces
        }

    def full_position_match(self, detected_fen: str, expected_fen: str) -> bool:
        """Check if the entire position matches"""
        detected_pos = detected_fen.split()[0]
        expected_pos = expected_fen.split()[0]
        return detected_pos == expected_pos

    def run_single_test(self, test_case: ChessTestCase, test_num: int) -> Dict:
        """Run a single test case"""
        print(f"\n{'=' * 60}")
        print(f"Test {test_num}: {test_case.image_path}")
        print(f"{'=' * 60}")

        response, processing_time = self.call_api(test_case)

        if 'error' in response:
            print(f"API Error: {response['error']}")
            return {
                'test_num': test_num,
                'image_path': str(test_case.image_path),
                'status': 'error',
                'error': response.get('error'),
                'processing_time': processing_time
            }

        detected_fen = response['fen']
        detected_board = response['board_matrix']

        occupancy_accuracy = self.calculate_occupancy_accuracy(detected_board, test_case.expected_fen)
        shape_accuracy = self.calculate_shape_accuracy(detected_board, test_case.expected_fen)
        piece_accuracy = self.calculate_piece_accuracy(detected_board, test_case.expected_fen)
        full_match = self.full_position_match(detected_fen, test_case.expected_fen)

        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Occupancy Accuracy: {occupancy_accuracy['accuracy']:.1f}%")
        print(f"Shape Accuracy: {shape_accuracy['accuracy']:.1f}%")
        print(f"Piece Accuracy: {piece_accuracy['accuracy']:.1f}%")
        print(f"Full Position Match: {'YES' if full_match else 'NO'}")
        print(f"Detected FEN: {detected_fen}")
        print(f"Expected FEN: {test_case.expected_fen}")

        if piece_accuracy['errors']:
            print(f"\nErrors ({len(piece_accuracy['errors'])}):")
            for error in piece_accuracy['errors'][:5]:
                print(f"  {error['position']}: detected '{error['detected']}', expected '{error['expected']}'")
            if len(piece_accuracy['errors']) > 5:
                print(f"  ... and {len(piece_accuracy['errors']) - 5} more")

        return {
            'test_num': test_num,
            'image_path': str(test_case.image_path),
            'status': 'success',
            'processing_time': processing_time,
            'occupancy_accuracy': occupancy_accuracy['accuracy'],
            'correct_occupancy': occupancy_accuracy['correct'],
            'incorrect_occupancy': occupancy_accuracy['incorrect'],
            'shape_accuracy': shape_accuracy['accuracy'],
            'correct_shapes': shape_accuracy['correct'],
            'incorrect_shapes': shape_accuracy['incorrect'],
            'piece_accuracy': piece_accuracy['accuracy'],
            'correct_pieces': piece_accuracy['correct'],
            'incorrect_pieces': piece_accuracy['incorrect'],
            'full_position_match': full_match,
            'detected_fen': detected_fen,
            'expected_fen': test_case.expected_fen,
            'errors': piece_accuracy['errors']
        }

    def run_all_tests(self, test_cases: List[ChessTestCase]) -> Dict:
        """Run all test cases"""
        print(f"\n{'#' * 60}")
        print(f"# Running Chess Detection API Test Suite")
        print(f"# Total Tests: {len(test_cases)}")
        print(f"# API URL: {self.api_url}")
        print(f"{'#' * 60}")

        self.results = []

        for i, test_case in enumerate(test_cases, 1):
            result = self.run_single_test(test_case, i)
            self.results.append(result)
            time.sleep(0.5)

        summary = self.generate_summary()
        self.print_summary(summary)
        self.save_results(summary)

        return summary

    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        successful_tests = [r for r in self.results if r['status'] == 'success']

        if not successful_tests:
            return {'error': 'No successful tests to analyze'}

        processing_times = [r['processing_time'] for r in successful_tests]
        occupancy_accuracies = [r['occupancy_accuracy'] for r in successful_tests]
        shape_accuracies = [r['shape_accuracy'] for r in successful_tests]
        piece_accuracies = [r['piece_accuracy'] for r in successful_tests]
        full_matches = [r['full_position_match'] for r in successful_tests]

        all_errors = []
        for test in successful_tests:
            all_errors.extend(test.get('errors', []))

        confusion_pairs = {}
        for error in all_errors:
            pair = (error['detected'], error['expected'])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        common_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total_tests': len(self.results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(self.results) - len(successful_tests),

            'performance': {
                'avg_processing_time': np.mean(processing_times),
                'min_processing_time': np.min(processing_times),
                'max_processing_time': np.max(processing_times),
                'p95_processing_time': np.percentile(processing_times, 95),
                'p99_processing_time': np.percentile(processing_times, 99),
            },

            'accuracy': {
                'avg_occupancy_accuracy': np.mean(occupancy_accuracies),
                'min_occupancy_accuracy': np.min(occupancy_accuracies),
                'max_occupancy_accuracy': np.max(occupancy_accuracies),
                'avg_shape_accuracy': np.mean(shape_accuracies),
                'min_shape_accuracy': np.min(shape_accuracies),
                'max_shape_accuracy': np.max(shape_accuracies),
                'avg_piece_accuracy': np.mean(piece_accuracies),
                'min_piece_accuracy': np.min(piece_accuracies),
                'max_piece_accuracy': np.max(piece_accuracies),
                'perfect_detection_rate': sum(1 for acc in piece_accuracies if acc == 100) / len(
                    piece_accuracies) * 100,
            },

            'position_matching': {
                'full_match_count': sum(full_matches),
                'full_match_rate': (sum(full_matches) / len(full_matches)) * 100,
                'partial_match_count': len(full_matches) - sum(full_matches),
            },

            'error_analysis': {
                'total_errors': len(all_errors),
                'common_confusions': [
                    {'detected': pair[0], 'expected': pair[1], 'count': count}
                    for pair, count in common_confusions
                ]
            }
        }

    def print_summary(self, summary: Dict):
        """Print formatted summary"""
        print(f"\n{'=' * 60}")
        print(f"TEST SUMMARY")
        print(f"{'=' * 60}")

        print(f"\n-> OVERALL: {summary['successful_tests']}/{summary['total_tests']} successful")

        perf = summary['performance']
        print(f"\n-> PERFORMANCE:")
        print(
            f"  Avg: {perf['avg_processing_time']:.2f}s | P95: {perf['p95_processing_time']:.2f}s | P99: {perf['p99_processing_time']:.2f}s")

        acc = summary['accuracy']
        print(f"\n-> OCCUPANCY ACCURACY:")
        print(
            f"  Avg: {acc['avg_occupancy_accuracy']:.1f}% | Min: {acc['min_occupancy_accuracy']:.1f}% | Max: {acc['max_occupancy_accuracy']:.1f}%")

        print(f"\n-> SHAPE ACCURACY:")
        print(
            f"  Avg: {acc['avg_shape_accuracy']:.1f}% | Min: {acc['min_shape_accuracy']:.1f}% | Max: {acc['max_shape_accuracy']:.1f}%")

        print(f"\n-> PIECE ACCURACY:")
        print(
            f"  Avg: {acc['avg_piece_accuracy']:.1f}% | Min: {acc['min_piece_accuracy']:.1f}% | Max: {acc['max_piece_accuracy']:.1f}%")
        print(f"  Perfect: {acc['perfect_detection_rate']:.1f}%")

        pos = summary['position_matching']
        print(
            f"\n-> FULL MATCH: {pos['full_match_count']}/{summary['successful_tests']} ({pos['full_match_rate']:.1f}%)")

        errors = summary['error_analysis']
        print(f"\n-> ERRORS: {errors['total_errors']} total")
        if errors['common_confusions']:
            print(f"  Top confusions:")
            for conf in errors['common_confusions'][:5]:
                print(f"    {conf['expected']} → {conf['detected']}: {conf['count']}x")

        print(f"\n{'=' * 60}\n")

    def save_results(self, summary: Dict):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_file = RESULTS_DIR / f"test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'summary': summary, 'detailed_results': self.results}, f, indent=2)
        print(f"-> Results: {results_file}")

        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = RESULTS_DIR / f"test_summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"-> CSV output: {csv_file}")


def main():
    """Main test execution"""
    if not TEST_DATA_DIR.exists():
        TEST_DATA_DIR.mkdir(exist_ok=True)
        print(f"-> Created {TEST_DATA_DIR}. Add test images and update TEST_CASES.")
        return

    tester = ChessAPITester(api_url=API_URL)

    try:
        health = requests.get(f"{API_URL}/", timeout=5)
        print(f"API is running at {API_URL}")
    except Exception as e:
        print(f"Cannot reach API: {e}")
        return

    summary = tester.run_all_tests(TEST_CASES)
    print("\n --- Test suite completed! ---")


if __name__ == "__main__":
    main()