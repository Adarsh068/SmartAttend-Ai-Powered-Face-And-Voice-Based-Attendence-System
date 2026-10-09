from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np 
import io
import librosa
import streamlit as st


@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)
        return embedding.tolist()
    except Exception as e:
        st.error('Voice recog error')
        return None
    

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
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.60,
                       window_sec=1.6, hop_sec=0.8):
    encoder = load_voice_encoder()
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    ids = list(candidates_dict.keys())
    stored = np.stack([_unit(candidates_dict[i]) for i in ids])

    win, hop = int(window_sec * sr), int(hop_sec * sr)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))

    loud_ref = np.percentile(np.abs(audio), 99) + 1e-9
    closest = {i: 0.0 for i in ids}
    present = {}

    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start:start + win]
        if np.sqrt(np.mean(chunk ** 2)) < 0.05 * loud_ref:
            continue
        try:
            wav = preprocess_wav(chunk)
            if len(wav) < int(0.5 * sr):
                continue
            emb = _unit(encoder.embed_utterance(wav))
        except Exception:
            continue

        sims = stored @ emb
        for i, s in zip(ids, sims):
            closest[i] = max(closest[i], float(s))

        k = int(np.argmax(sims))
        if sims[k] >= threshold:
            present[ids[k]] = max(present.get(ids[k], 0.0), float(sims[k]))

    return present, closest