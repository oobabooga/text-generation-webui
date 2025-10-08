"""Flask API Blueprint for agent learning progress endpoint."""
from flask import Blueprint, jsonify
from .learning_metrics import metrics

# Create Blueprint for learning API
learning_bp = Blueprint('learning', __name__, url_prefix='/learning')


@learning_bp.route('/get_learning_progress', methods=['GET'])
def get_learning_progress():
    """
    GET /learning/get_learning_progress
    
    Returns the current learning metrics and progress.
    
    Response format:
    {
        "success_rate": [0.0, 1.0, ...],
        "reaction_time": [0.5, 0.8, ...],
        "logic": [1.0, 0.9, ...],
        "memory": [1.0, 1.0, ...],
        "creativity": [0.8, 0.9, ...],
        "progress": 75.5,
        "last_update_ts": 1234567890.123,
        "version": 1
    }
    """
    try:
        current_metrics = metrics.get_metrics()
        return jsonify(current_metrics), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to retrieve learning progress"
        }), 500
