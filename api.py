import os
import json
import tempfile
from typing import Tuple, Optional
from flask_login import current_user
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import speech_recognition as sr
import boto3
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()

api = Blueprint("api", __name__)

# --- Config ---
ALLOWED_EXTENSIONS = {"wav", "flac", "aif", "aiff", "aifc"}
MAX_FILE_MB = 25  # guardrail

def get_sagemaker_client():
    region = os.getenv("AWS_REGION")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return boto3.client(
        "sagemaker-runtime",
        region_name=region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )

def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_wav(path: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    # You can swap to r.recognize_sphinx(audio) if you need offline
    return r.recognize_google(audio)

def parse_model_output(raw: str) -> Tuple[str, Optional[float]]:
    """
    Your current code expects "pred,conf" as a string.
    This handles JSON or "pred,conf" or a bare label.
    """
    s = raw.strip()
    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            pred = str(obj.get("prediction", "")).strip()
            conf = obj.get("confidence")
            return pred, float(conf) if conf is not None else None
        if isinstance(obj, list) and obj:
            # e.g., ["1", 98.2]
            pred = str(obj[0]).strip()
            conf = obj[1] if len(obj) > 1 else None
            return pred, float(conf) if conf is not None else None
    except json.JSONDecodeError:
        pass

    # Try CSV "pred,conf"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        pred = parts[0]
        try:
            conf = float(parts[1])
        except Exception:
            conf = None
        return pred, conf

    # Fallback: just a label
    return s, None

def label_text(pred: str) -> str:
    p = pred.strip().lower()
    if p in {"1", "true", "dementia", "positive", "likely_dementia"}:
        return "Likely Dementia"
    return "Not Dementia"

@api.route("/api/predict", methods=["POST"])
def predict():
    try:
        file_storage = request.files.get("audio")
        
        # Validate small things early
        if not file_storage:
            return jsonify({"error": "Missing audio. Provide 'audio' file or 'audio_base64'."}), 400

        # Validate fields
        sex = 1 if current_user.sex == 'M' else 0
        age = int(current_user.age)
        mmse = current_user.mmse_score

        # Prepare a temp file for the transcription lib
        if file_storage:
            if file_storage.filename == "":
                return jsonify({"error": "Empty file name."}), 400
            if not allowed_file(file_storage.filename):
                return jsonify({"error": f"Unsupported audio type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 415

            file_storage.seek(0, os.SEEK_END)
            size_mb = file_storage.tell() / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                return jsonify({"error": f"File too large (> {MAX_FILE_MB} MB)."}), 413
            file_storage.seek(0)

            suffix = "." + file_storage.filename.rsplit(".", 1)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                file_storage.save(tmp_path)
        else:
            return jsonify({"error": "Base64 upload not implemented in this example."}), 400

        # Transcribe
        try:
            transcript = transcribe_wav(tmp_path)
        except sr.UnknownValueError:
            os.unlink(tmp_path)
            return jsonify({"error": "Could not understand audio (speech recognition)."}), 422
        except sr.RequestError as e:
            os.unlink(tmp_path)
            return jsonify({"error": f"Speech recognition service error: {e}"}), 502
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Build payload for your model
        endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "canvas-Dementia-Deployment")
        payload = {
            "data": {
                "features": {
                    "values": [[sex, age, mmse, transcript]]
                }
            }
        }
        body = json.dumps(payload).encode("utf-8")

        # Call SageMaker
        try:
            client = get_sagemaker_client()
            resp = client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=body,
            )
            raw = resp["Body"].read().decode("utf-8")
        except (BotoCoreError, ClientError) as e:
            return jsonify({"error": f"SageMaker error: {str(e)}"}), 502

        pred_raw, conf = parse_model_output(raw)
        result_label = label_text(pred_raw)

        return jsonify({
            "ok": True,
            "endpoint": endpoint_name,
            "inputs": {"sex": sex, "age": age, "mmse": mmse},
            "transcript": transcript,
            "model_raw": raw,
            "prediction": result_label,
            "confidence": conf  # may be None if the model didn't return it parsably
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Don’t leak internals in prod; log instead
        if current_app.debug:
            return jsonify({"error": f"Unhandled error: {repr(e)}"}), 500
        return jsonify({"error": "Internal error"}), 500