import customtkinter as ctk
import subprocess, os, threading, json
import sys
import sys, os, importlib.util

if sys.platform == "win32":
    for pkg in ("cublas", "cudnn", "cuda_runtime"):
        spec = importlib.util.find_spec(f"nvidia.{pkg}")
        if spec and spec.submodule_search_locations:
            bin_dir = os.path.join(spec.submodule_search_locations[0], "bin")
            if os.path.isdir(bin_dir):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]   
def get_tool(name):
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, name + ".exe")
    return path if os.path.exists(path) else name

FFMPEG = get_tool("ffmpeg")
FFPROBE = get_tool("ffprobe")

ctk.set_appearance_mode("dark")

class SubtitleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Subtitle Generator")
        self.geometry("600x500")

        self.video_path = None
        self.audio_tracks = []
        self.model = None
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="AI Subtitle Generator", font=("Arial", 20, "bold")).pack(pady=10)

        file_frame = ctk.CTkFrame(self)
        file_frame.pack(pady=5, padx=20, fill="x")

        self.file_label = ctk.CTkLabel(file_frame, text="No file selected")
        self.file_label.pack(side="left", padx=10)

        ctk.CTkButton(file_frame, text="Browse", width=100, command=self._pick_file).pack(side="right", padx=10)

        # Language selector
        lang_frame = ctk.CTkFrame(self)
        lang_frame.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(lang_frame, text="Language:").pack(side="left", padx=10)
        self.lang_var = ctk.StringVar(value="English")
        ctk.CTkOptionMenu(lang_frame, values=["English", "Japanese", "Spanish", "French", "German", "Auto"],
                          variable=self.lang_var, width=120).pack(side="left", padx=10)

        # Audio track selector
        track_frame = ctk.CTkFrame(self)
        track_frame.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(track_frame, text="Audio Track:").pack(side="left", padx=10)
        self.track_var = ctk.StringVar(value="0")
        self.track_menu = ctk.CTkOptionMenu(track_frame, values=["0"], variable=self.track_var, width=120)
        self.track_menu.pack(side="left", padx=10)

        self.gen_btn = ctk.CTkButton(self, text="Generate Subtitles", command=self._generate, height=40)
        self.gen_btn.pack(pady=10)

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self.progress.pack(pady=5, padx=20, fill="x")

        self.output = ctk.CTkTextbox(self)
        self.output.pack(pady=10, padx=20, fill="both", expand=True)

        ctk.CTkButton(self, text="Save .srt", command=self._save_srt).pack(pady=5)

    def _pick_file(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")])
        if path:
            self.video_path = path
            self.file_label.configure(text=os.path.basename(path))
            self._scan_tracks()

    def _scan_tracks(self):
        try:
            result = subprocess.run(
                [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_streams", self.video_path],
                capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout)
            tracks = []
            audio_idx = 0
            for stream in data["streams"]:
                if stream.get("codec_type") == "audio":
                    lang = stream.get("tags", {}).get("language", "und")
                    tracks.append(f"{audio_idx} [{lang}]")
                    audio_idx += 1
            if tracks:
                self.audio_tracks = tracks
                self.track_menu.configure(values=tracks)
                self.track_var.set(tracks[0])
        except Exception as e:
            self.track_menu.configure(values=["0"])
            self.track_var.set("0")

    def _get_track_index(self):
        return self.track_var.get().split(" ")[0]

    def _get_language(self):
        mapping = {
            "English": "en", "Japanese": "ja", "Spanish": "es",
            "French": "fr", "German": "de", "Auto": None
        }
        return mapping.get(self.lang_var.get(), None)

    def _get_prompt(self):
        lang = self.lang_var.get()
        if lang == "Auto":
            return None
        prompts = {
            "English": "This is a transcript of English dialogue. Use proper punctuation and capitalization.",
            "Japanese": "This is a transcript of Japanese dialogue.",
            "Spanish": "This is a transcript of Spanish dialogue. Use proper punctuation and capitalization.",
            "French": "This is a transcript of French dialogue. Use proper punctuation and capitalization.",
            "German": "This is a transcript of German dialogue. Use proper punctuation and capitalization.",
        }
        return prompts.get(lang)

    def _get_model(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float32")   
        return self.model

    def _generate(self):
        if not self.video_path:
            return
        self.gen_btn.configure(state="disabled")
        self.progress.start()
        threading.Thread(target=self._run_whisper, daemon=True).start()

    def _run_whisper(self):
        audio = None
        try:
            audio = self.video_path.rsplit(".", 1)[0] + "_audio.wav"
            track = self._get_track_index()

            subprocess.run([FFMPEG, "-i", self.video_path,
                        "-map", f"0:a:{track}",
                        "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", "-y", audio], check=True)

            model = self._get_model()
            segments, info = model.transcribe(
                audio,
                language=self._get_language(),
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=300,
                ),
                condition_on_previous_text=False,
                no_speech_threshold=0.3,
                initial_prompt=self._get_prompt(),
            )

            lines = []
            for i, seg in enumerate(segments, 1):
                start = self._ts(seg.start)
                end = self._ts(seg.end)
                lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
            srt = "\n".join(lines)

            self.after(0, self._on_done, srt)

        except Exception as e:
            self.after(0, self._on_error, str(e))
        finally:
            if audio and os.path.exists(audio):
                os.remove(audio)

    def _on_done(self, srt):
        self.output.delete("1.0", "end")
        self.output.insert("1.0", srt)
        self.progress.stop()
        self.gen_btn.configure(state="normal")

    def _on_error(self, msg):
        self.output.delete("1.0", "end")
        self.output.insert("1.0", f"Error: {msg}")
        self.progress.stop()
        self.gen_btn.configure(state="normal")

    def _save_srt(self):
        import tkinter.filedialog as fd
        path = fd.asksaveasfilename(
            defaultextension=".srt",
            filetypes=[("SRT", "*.srt")],
            initialfile="subtitles.srt"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.output.get("1.0", "end").strip())

    @staticmethod
    def _ts(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

app = SubtitleApp()
app.mainloop()   