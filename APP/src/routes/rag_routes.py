import uuid
import threading

from flask import Blueprint, jsonify, request

from ..assistant.custom_assistant import get_multi_agent_response

rag_bp = Blueprint('rag', __name__, url_prefix='/api/rag')


@rag_bp.route('/query', methods=['POST'])
def query_rag():
    question = request.get_json().get('question')
    if not question:
        return jsonify({
            "code": 400,
            "message": "Question parameter is required"
        }), 400
    else:
        multi_agent_response = get_multi_agent_response(question)
        return jsonify({'message': multi_agent_response})


def async_task(task_id):
    pass


@rag_bp.route('/build', methods=['POST'])
def build_rag():
    task_id = str(uuid.uuid4())

    threading.Thread(target=async_task, args=task_id).start()

    return jsonify({'task_id': task_id}), 202
