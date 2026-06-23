
import os, io, json, tempfile, base64, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from flask import Flask, request, jsonify, render_template

TARGET_SR    = 16000
MAX_DURATION = 5.5
MAX_LENGTH   = int(TARGET_SR * MAX_DURATION)

LABELS_A = [
    'Qalqalah_Benar', 'Qalqalah_Tidak_Memantul', 'Qalqalah_Berlebih',
    'Mad_Thabii_Benar', 'Mad_Thabii_Kurang', 'Mad_Thabii_Berlebih',
    'Mad_Wajib_Benar', 'Mad_Wajib_Kurang', 'Mad_Wajib_Berlebih',
    'Mad_Lazim_Benar', 'Mad_Lazim_Kurang',
]

LABELS_B = [
    'Idzhar', 'Idgham_Bighunnah', 'Idgham_Bilaghunnah',
    'Iqlab', 'Ikhfa',
]

NUM_CLASSES_A = len(LABELS_A)
NUM_CLASSES_B = len(LABELS_B)

MODEL_NAME   = 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic'
MODEL_A_PATH = 'models/wav2vec_freeze6.pth'
MODEL_B_PATH = 'models/nun_sukun_model.pth'

WHISPER_MODEL_SIZE = 'tiny'  
ENERGY_THRESHOLD   = -5.0   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model A  - transformer
from transformers import Wav2Vec2Model

class TajwidTransformer(nn.Module):
    def __init__(self, model_name, num_classes, freeze_layers=6):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False
        for i in range(min(freeze_layers, 24)):
            for p in self.wav2vec2.encoder.layers[i].parameters():
                p.requires_grad = False
        h = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.wav2vec2(x).last_hidden_state.mean(dim=1)
        return self.classifier(out)


# Model B - CNN masih dummy, ganti ajaa

class DummyCNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            TARGET_SR, n_fft=512, hop_length=160, n_mels=128)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(64 * 32, 128, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        mel = self.mel(x).unsqueeze(1)
        c   = self.cnn(mel)
        b, ch, fr, t = c.size()
        c   = c.permute(0, 3, 1, 2).contiguous().view(b, t, ch * fr)
        h, _ = self.lstm(c)
        return self.fc(h.mean(dim=1))


def preprocess_segment(audio_np, sr):
    waveform = torch.from_numpy(audio_np).float()
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    mx = waveform.abs().max()
    if mx > 0:
        waveform = waveform / mx
    if len(waveform) > MAX_LENGTH:
        start = (len(waveform) - MAX_LENGTH) // 2
        waveform = waveform[start:start + MAX_LENGTH]
    elif len(waveform) < MAX_LENGTH:
        waveform = F.pad(waveform, (0, MAX_LENGTH - len(waveform)))
    return waveform.unsqueeze(0).to(device)


def segment_to_base64(audio_np, sr):
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


whisper_model = None

def load_whisper():
    global whisper_model
    if whisper_model is None:
        import whisper
        print(f'  Memuat Whisper {WHISPER_MODEL_SIZE}...')
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        print(f'  Whisper siap.')
    return whisper_model


def segment_with_whisper(audio_path, min_dur=0.2, max_dur=3.5):
    import whisper

    model = load_whisper()

    audio_np, sr = sf.read(audio_path, dtype='float32')
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != TARGET_SR:
        audio_np = torchaudio.functional.resample(
            torch.from_numpy(audio_np).float(), sr, TARGET_SR
        ).numpy()

    result = model.transcribe(
        audio_np,
        language='ar',
        word_timestamps=True,
        fp16=(device.type == 'cuda'),
    )


    segments = []
    for seg in result.get('segments', []):
        for word in seg.get('words', []):
            s = word['start']
            e = word['end']
            dur = e - s
            text = word.get('word', '').strip()

            if dur < min_dur:
                continue

            s_idx = int(s * TARGET_SR)
            e_idx = int(e * TARGET_SR)
            chunk = audio_np[max(0, s_idx):min(len(audio_np), e_idx)]

            if len(chunk) < int(min_dur * TARGET_SR):
                continue

            # Split jika terlalu panjang
            if dur > max_dur:
                sub_len = int(max_dur * TARGET_SR)
                hop     = int(max_dur * 0.5 * TARGET_SR)
                pos     = 0
                while pos < len(chunk):
                    sub = chunk[pos:pos + sub_len]
                    if len(sub) > int(min_dur * TARGET_SR):
                        segments.append({
                            'start': round(s + pos / TARGET_SR, 3),
                            'end': round(s + (pos + len(sub)) / TARGET_SR, 3),
                            'duration': round(len(sub) / TARGET_SR, 3),
                            'text': text if pos == 0 else '',
                            'audio': sub,
                        })
                    pos += hop
            else:
                segments.append({
                    'start': round(s, 3),
                    'end': round(e, 3),
                    'duration': round(dur, 3),
                    'text': text,
                    'audio': chunk,
                })

    return segments


def segment_with_energy(audio_np, sr,
                        min_dur=0.25, max_dur=3.5,
                        min_silence_ms=80, frame_ms=20, hop_ms=5,
                        threshold_percentile=35, smooth_kernel=5):
    if sr != TARGET_SR:
        audio_np = torchaudio.functional.resample(
            torch.from_numpy(audio_np).float(), sr, TARGET_SR
        ).numpy()
        sr = TARGET_SR

    frame_len = int(frame_ms / 1000 * sr)
    hop_len   = int(hop_ms / 1000 * sr)
    n_frames  = max(1, (len(audio_np) - frame_len) // hop_len)

    energy_db = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio_np[start:start + frame_len]
        rms   = np.sqrt(np.mean(frame ** 2) + 1e-10)
        energy_db[i] = 20 * np.log10(rms + 1e-10)

    energy_smooth = uniform_filter1d(energy_db, size=smooth_kernel)
    threshold = np.percentile(energy_smooth, threshold_percentile)
    is_speech = energy_smooth > threshold

    min_silence_frames = int(min_silence_ms / hop_ms)
    min_dur_frames     = int(min_dur * 1000 / hop_ms)

    raw_segments = []
    in_speech = False
    seg_start = 0
    silence_count = 0

    for i in range(len(is_speech)):
        if is_speech[i]:
            if not in_speech:
                seg_start = i
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    seg_end = i - silence_count
                    if seg_end - seg_start >= min_dur_frames:
                        s_sec = seg_start * hop_len / sr
                        e_sec = seg_end * hop_len / sr
                        raw_segments.append((s_sec, e_sec))
                    in_speech = False
                    silence_count = 0

    if in_speech:
        seg_end = len(is_speech) - 1
        if seg_end - seg_start >= min_dur_frames:
            raw_segments.append((seg_start * hop_len / sr, seg_end * hop_len / sr))

    result = []
    for s_sec, e_sec in raw_segments:
        dur   = e_sec - s_sec
        s_idx = int(s_sec * sr)
        e_idx = int(e_sec * sr)
        chunk = audio_np[max(0, s_idx):min(len(audio_np), e_idx)]

        if dur > max_dur:
            sub_len = int(max_dur * sr)
            hop     = int(max_dur * 0.5 * sr)
            pos     = 0
            while pos < len(chunk):
                sub = chunk[pos:pos + sub_len]
                if len(sub) > int(min_dur * sr):
                    result.append({
                        'start': round(s_sec + pos / sr, 3),
                        'end': round(s_sec + (pos + len(sub)) / sr, 3),
                        'duration': round(len(sub) / sr, 3),
                        'text': '',
                        'audio': sub,
                    })
                pos += hop
        elif dur >= min_dur:
            result.append({
                'start': round(s_sec, 3),
                'end': round(e_sec, 3),
                'duration': round(dur, 3),
                'text': '',
                'audio': chunk,
            })

    return result



def segment_with_sliding_window(audio_np, sr,
                                window_dur=2.0, hop_dur=1.0,
                                skip_silence=True,
                                frame_ms=20, hop_ms=5,
                                threshold_percentile=35, smooth_kernel=5):
    if sr != TARGET_SR:
        audio_np = torchaudio.functional.resample(
            torch.from_numpy(audio_np).float(), sr, TARGET_SR
        ).numpy()
        sr = TARGET_SR

    win_len = int(window_dur * sr)
    hop_len = int(hop_dur * sr)

    # Hitung threshold energy global (sama gaya dgn segment_with_energy)
    silence_thresh_db = None
    if skip_silence:
        f_len = int(frame_ms / 1000 * sr)
        f_hop = int(hop_ms / 1000 * sr)
        n_frames = max(1, (len(audio_np) - f_len) // f_hop)
        energy_db = np.zeros(n_frames)
        for i in range(n_frames):
            st = i * f_hop
            frame = audio_np[st:st + f_len]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            energy_db[i] = 20 * np.log10(rms + 1e-10)
        energy_smooth = uniform_filter1d(energy_db, size=smooth_kernel)
        silence_thresh_db = np.percentile(energy_smooth, threshold_percentile)

    result = []
    pos = 0
    while pos < len(audio_np):
        chunk = audio_np[pos:pos + win_len]
        if len(chunk) < int(0.2 * sr):  # buang sisa terlalu pendek
            break

        if skip_silence and silence_thresh_db is not None:
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-10)
            chunk_db = 20 * np.log10(rms + 1e-10)
            if chunk_db <= silence_thresh_db:
                pos += hop_len
                continue

        s_sec = pos / sr
        e_sec = (pos + len(chunk)) / sr
        result.append({
            'start': round(s_sec, 3),
            'end': round(e_sec, 3),
            'duration': round(len(chunk) / sr, 3),
            'text': '',
            'audio': chunk,
        })
        pos += hop_len

    return result


def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1).item()


def predict_segment(audio_np, sr, model_a, model_b):
    tensor = preprocess_segment(audio_np, sr)

    with torch.no_grad():
        # Model A
        logits_a = model_a(tensor)
        prob_a   = F.softmax(logits_a, dim=1)[0]
        pred_a   = prob_a.argmax().item()
        conf_a   = prob_a.max().item()
        e_a      = energy_score(logits_a)
        e_norm_a = e_a / np.log(NUM_CLASSES_A)

        # Model B
        logits_b = model_b(tensor)
        prob_b   = F.softmax(logits_b, dim=1)[0]
        pred_b   = prob_b.argmax().item()
        conf_b   = prob_b.max().item()
        e_b      = energy_score(logits_b)
        e_norm_b = e_b / np.log(NUM_CLASSES_B)

    # Energy lebih rendah = lebih yakin
    if e_norm_a <= e_norm_b and e_a < ENERGY_THRESHOLD:
        return {
            'model': 'A',
            'domain': 'Qalqalah / Mad',
            'label': LABELS_A[pred_a],
            'confidence': round(conf_a * 100, 1),
            'energy': round(e_a, 2),
        }
    elif e_b < ENERGY_THRESHOLD:
        return {
            'model': 'B',
            'domain': 'Nun Sukun / Tanwin',
            'label': LABELS_B[pred_b],
            'confidence': round(conf_b * 100, 1),
            'energy': round(e_b, 2),
        }
    else:
        return {
            'model': '-',
            'domain': 'Tidak terdeteksi',
            'label': 'Tidak ada hukum tajwid',
            'confidence': 0,
            'energy': 0,
        }


#App Flask

app = Flask(__name__)
model_a = None
model_b = None


def load_models():
    global model_a, model_b

    if model_a is None:
        print('Memuat Model A (Wav2Vec 2.0)...')
        if os.path.exists(MODEL_A_PATH):
            model_a = TajwidTransformer(MODEL_NAME, NUM_CLASSES_A, freeze_layers=6).to(device)
            model_a.load_state_dict(
                torch.load(MODEL_A_PATH, map_location=device, mmap=True))
            model_a.eval()
            print('  Model A siap.')
        else:
            print(f'  {MODEL_A_PATH} tidak ditemukan.')

    if model_b is None:
        print('Memuat Model B (CNN-BiLSTM / Dummy)...')
        if os.path.exists(MODEL_B_PATH):
            # GANTI kode ini ke model nun tanwin
            pass
        if model_b is None:
            print('  Model B: DUMMY (random output)')
            model_b = DummyCNNBiLSTM(NUM_CLASSES_B).to(device)
            model_b.eval()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'Tidak ada file audio'}), 400

    method = request.form.get('method', 'whisper')  # whisper atau energy

    audio_file = request.files['audio']
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_file.save(tmp.name)
    tmp.close()

    try:
        audio_np, sr = sf.read(tmp.name, dtype='float32')
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        total_duration = len(audio_np) / sr

        # Segmentasi
        segments = []
        used_method = method

        if method == 'whisper':
            try:
                segments = segment_with_whisper(tmp.name)
                used_method = 'whisper'
            except Exception as e:
                print(f'Whisper gagal: {e}, kembali ke Energy VAD')
                segments = segment_with_energy(audio_np, sr)
                used_method = 'energy'
        elif method == 'sliding':
            segments = segment_with_sliding_window(audio_np, sr)
            used_method = 'sliding'
        else:
            segments = segment_with_energy(audio_np, sr)
            used_method = 'energy'

        if not segments:
            return jsonify({
                'total_duration': round(total_duration, 2),
                'num_segments': 0,
                'method': used_method,
                'segments': [],
                'message': 'Tidak ditemukan segmen speech.'
            })

        # Prediksi per segmen
        results = []
        for i, seg in enumerate(segments):
            pred = predict_segment(seg['audio'], TARGET_SR, model_a, model_b)
            results.append({
                'index': i + 1,
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['duration'],
                'text': seg.get('text', ''),
                'audio_b64': segment_to_base64(seg['audio'], TARGET_SR),
                **pred,
            })

        return jsonify({
            'total_duration': round(total_duration, 2),
            'num_segments': len(results),
            'method': used_method,
            'segments': results,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp.name)


if __name__ == '__main__':
    try:
        print('Loading models...')
        load_models()
        print('Models loaded.')
    except Exception as e:
        print(f'ERROR: {e}')
        traceback.print_exc()

    try:
        print('Loading Whisper...')
        load_whisper()
    except Exception as e:
        print(f'Whisper tidak ada: {e}')

    print(f'\nDevice : {device}')
    print(f'Model A: {NUM_CLASSES_A} kelas (Qalqalah + Mad)')
    print(f'Model B: {NUM_CLASSES_B} kelas (Nun Sukun + Tanwin)')
    print(f'\nApp siap -> http://localhost:5000\n')
    app.run(debug=False, host='0.0.0.0', port=5000)