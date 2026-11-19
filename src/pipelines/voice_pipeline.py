from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np 
import io
import json
import librosa
import streamlit as st


@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    try:
        encoder = load_voice_encoder()
        audio, sr = _load_audio(audio_bytes)
        if len(audio) < int(0.5 * sr):
            raise ValueError('record at least half a second of speech')
        wav = preprocess_wav(audio)
        if len(wav) < int(0.5 * sr):
            raise ValueError('no usable speech was found in the recording')
        embedding = encoder.embed_utterance(wav)
        return embedding.tolist()
    except Exception as e:
        raise RuntimeError(f'voice enrollment failed: {e}') from e
    

def identify_speaker(new_embedding, candidates_dict, threshold=0.65):
    if new_embedding is None or not candidates_dict:
        return None, 0.0
    
    best_sid = None
    best_score = -1.0

    for sid, stored_embedding in candidates_dict.items():
        if stored_embedding:
            similarity = np.dot(new_embedding, stored_embedding)
            if similarity > best_score:
                best_score = similarity
                best_sid = sid

    if best_score >= threshold:
        return best_sid, best_score
    
    return None, best_score


def _unit(v):
    if isinstance(v, str):
        v = json.loads(v)
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _load_audio(audio_bytes):
    if not audio_bytes:
        raise ValueError('the recording is empty')
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('the recording contains no valid audio')
    return audio, sr


def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.60,
                       window_sec=1.6, hop_sec=0.8):
    encoder = load_voice_encoder()
    audio, sr = _load_audio(audio_bytes)

    valid_candidates = {}
    for sid, embedding in candidates_dict.items():
        try:
            candidate = _unit(embedding)
            if candidate.ndim == 1 and candidate.size and np.linalg.norm(candidate) > 0:
                valid_candidates[sid] = candidate
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if not valid_candidates:
        raise ValueError('no valid enrolled voice profiles were found')

    ids = list(valid_candidates.keys())
    stored = np.stack([valid_candidates[i] for i in ids])

    win, hop = int(window_sec * sr), int(hop_sec * sr)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))

    loud_ref = np.percentile(np.abs(audio), 99) + 1e-9
    closest = {i: 0.0 for i in ids}
    present = {}
    processed_chunks = 0

    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start:start + win]
        if np.sqrt(np.mean(chunk ** 2)) < 0.02 * loud_ref:
            continue
        try:
            wav = preprocess_wav(chunk)
            if len(wav) < int(0.5 * sr):
                continue
            emb = _unit(encoder.embed_utterance(wav))
            if not emb.size or np.linalg.norm(emb) == 0:
                continue
        except (RuntimeError, ValueError):
            continue
        processed_chunks += 1

        sims = stored @ emb
        for i, s in zip(ids, sims):
            closest[i] = max(closest[i], float(s))

        k = int(np.argmax(sims))
        if sims[k] >= threshold:
            present[ids[k]] = max(present.get(ids[k], 0.0), float(sims[k]))

    if not processed_chunks:
        raise ValueError('no usable speech was found in the recording')
    return present, closest