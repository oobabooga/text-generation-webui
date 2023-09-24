import os
import whisperx
import gc
import tempfile
import shutil


def transcribe_align_diarize(filename, file_data, device, batch_size, compute_type, model_name, language, hf_token):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as tmp_file:
        shutil.copyfileobj(file_data, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        # Load the audio file first
        audio = whisperx.load_audio(tmp_file_path)

        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model(
            model_name, device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=batch_size)
        gc.collect()
        del model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=language or result["language"], device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        gc.collect()
        del model_a

        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        gc.collect()
        del diarize_model

    finally:
        # Delete the temporary file
        os.unlink(tmp_file_path)

    return result
