#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STM32 handwriting result receiver.

Run on the PC before powering/resetting the STM32 board:
    python web_receiver.py

- ESP8266/STM32 sends JSON lines to TCP port 8000.
- Browser opens http://127.0.0.1:8080/ to view latest results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import socketserver
import sys
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {
    "latest": None,
    "history": [],
    "tcp_clients": 0,
    "updated_at": None,
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}

TRANSLATE_MODES = {"translate", "word"}
PC_CNN_MODES = {"pc_cnn"}
BAIDU_API = "https://fanyi-api.baidu.com/api/trans/vip/translate"
BAIDU_TIMEOUT_S = 2.5
TTS_ENABLED = True
DEEPSEEK_ENABLED = False
DEEPSEEK_TRANSLATOR: "DeepSeekTranslator | None" = None
DEEPSEEK_URL = "https://chat.deepseek.com/"
DEEPSEEK_PROFILE_DIR = Path(__file__).with_name("本机浏览器登录缓存")
DEEPSEEK_SYSTEM_PROMPT = (
    "You are an English to Chinese translation helper. "
    "For every later input, output exactly one short line: "
    "CN=<actual Chinese characters>; PY=<actual pinyin with spaces>. "
    "Do not echo the input. Do not output placeholders, examples, numbering, or explanations."
)
PC_CNN_CODE_DIR = Path(__file__).resolve().parents[1] / "训练与模型工具"
PC_CNN_MODEL_PATH = PC_CNN_CODE_DIR / "models" / "cnn_emnist_byclass.pth"
PC_CNN_NORMAL_SIZE = 20
PC_CNN_GRID_SIZE = 28
PC_CNN_MIN_SEGMENT_W = 2
PC_CNN_MAX_SEGMENTS = 12
PC_CNN_LOCK = threading.Lock()
PC_CNN_CACHE: dict[str, Any] = {"model": None, "device": None, "error": None, "loaded": False}

# Offline fallback for mode 4. Keep this small and dependency-free so the demo
# still works when the PC has no internet or cloud credentials.
LOCAL_WORDS: dict[str, str] = {
    "a": "\u4e00\u4e2a",
    "an": "\u4e00\u4e2a",
    "and": "\u548c",
    "apple": "\u82f9\u679c",
    "banana": "\u9999\u8549",
    "book": "\u4e66",
    "boy": "\u7537\u5b69",
    "cat": "\u732b",
    "china": "\u4e2d\u56fd",
    "computer": "\u7535\u8111",
    "dog": "\u72d7",
    "english": "\u82f1\u8bed",
    "girl": "\u5973\u5b69",
    "good": "\u597d\u7684",
    "hello": "\u4f60\u597d",
    "hi": "\u4f60\u597d",
    "is": "\u662f",
    "love": "\u7231",
    "man": "\u7537\u4eba",
    "moon": "\u6708\u4eae",
    "morning": "\u65e9\u6668",
    "name": "\u540d\u5b57",
    "night": "\u591c\u665a",
    "one": "\u4e00",
    "orange": "\u6a59\u5b50",
    "pc": "\u7535\u8111",
    "pen": "\u94a2\u7b14",
    "read": "\u9605\u8bfb",
    "red": "\u7ea2\u8272",
    "run": "\u8dd1\u6b65",
    "stm32": "STM32 \u5355\u7247\u673a",
    "sun": "\u592a\u9633",
    "teacher": "\u8001\u5e08",
    "test": "\u6d4b\u8bd5",
    "the": "\u8fd9\u4e2a",
    "water": "\u6c34",
    "wifi": "\u65e0\u7ebf\u7f51\u7edc",
    "woman": "\u5973\u4eba",
    "word": "\u5355\u8bcd",
    "world": "\u4e16\u754c",
    "write": "\u4e66\u5199",
    "yes": "\u662f\u7684",
}

ASCII_TRANSLATIONS: dict[str, str] = {
    "a": "yi ge",
    "an": "yi ge",
    "and": "he",
    "apple": "ping guo",
    "banana": "xiang jiao",
    "book": "shu",
    "boy": "nan hai",
    "cat": "mao",
    "china": "zhong guo",
    "computer": "dian nao",
    "dog": "gou",
    "english": "ying yu",
    "girl": "nv hai",
    "good": "hao de",
    "hello": "ni hao",
    "hi": "ni hao",
    "is": "shi",
    "love": "ai",
    "man": "nan ren",
    "moon": "yue liang",
    "morning": "zao chen",
    "name": "ming zi",
    "night": "ye wan",
    "one": "yi",
    "orange": "cheng zi",
    "pc": "dian nao",
    "pen": "gang bi",
    "read": "yue du",
    "red": "hong se",
    "run": "pao bu",
    "stm32": "STM32 dan pian ji",
    "sun": "tai yang",
    "teacher": "lao shi",
    "test": "ce shi",
    "the": "zhe ge",
    "water": "shui",
    "wifi": "wu xian wang luo",
    "woman": "nv ren",
    "word": "dan ci",
    "world": "shi jie",
    "write": "shu xie",
    "yes": "shi de",
}

ZH_CODE_TRANSLATIONS: dict[str, str] = {
    "hi": "4F60,597D",
    "hello": "4F60,597D",
    "read": "9605,8BFB",
    "book": "4E66",
    "write": "5199",
    "run": "8DD1,6B65",
}

def add_event(item: dict[str, Any]) -> None:
    item.setdefault("received_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    with STATE_LOCK:
        STATE["latest"] = item
        STATE["updated_at"] = item["received_at"]
        STATE["history"].append(item)
        STATE["history"] = STATE["history"][-200:]


def clean_word(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"[A-Za-z][A-Za-z'\-]*", text)
    return match.group(0).lower() if match else text[:32]


def local_translate(word: str) -> tuple[str, str]:
    if not word:
        return "\u672a\u8bc6\u522b\u5230\u82f1\u6587\u5355\u8bcd", "local-empty"
    if word in LOCAL_WORDS:
        return LOCAL_WORDS[word], "local-dict"
    if word.endswith("ing") and len(word) > 5:
        return f"姝ｅ湪{word[:-3]}", "local-rule"
    if word.endswith("ed") and len(word) > 4:
        return f"{word[:-2]}鐨勮繃鍘诲紡/杩囧幓鍒嗚瘝", "local-rule"
    if word.endswith("s") and len(word) > 3:
        base = word[:-1]
        if base in LOCAL_WORDS:
            return f"{LOCAL_WORDS[base]}\uff08\u590d\u6570\uff09", "local-rule"
    return f"{word}\uff08\u6682\u65e0\u79bb\u7ebf\u8bcd\u5178\u91ca\u4e49\uff09", "local-unknown"


def baidu_translate(word: str) -> tuple[str, str] | None:
    app_id = os.environ.get("BAIDU_TRANSLATE_APP_ID") or os.environ.get("BAIDU_APP_ID")
    secret = os.environ.get("BAIDU_TRANSLATE_SECRET") or os.environ.get("BAIDU_SECRET_KEY")
    if not app_id or not secret or not word:
        return None

    salt = str(int(time.time() * 1000))
    sign_src = f"{app_id}{word}{salt}{secret}"
    sign = hashlib.md5(sign_src.encode("utf-8")).hexdigest()
    query = urlencode(
        {
            "q": word,
            "from": "en",
            "to": "zh",
            "appid": app_id,
            "salt": salt,
            "sign": sign,
        }
    )
    try:
        with urlopen(f"{BAIDU_API}?{query}", timeout=BAIDU_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"[TRANSLATE] baidu unavailable: {exc}")
        return None

    results = body.get("trans_result") or []
    if results and isinstance(results[0], dict) and results[0].get("dst"):
        return str(results[0]["dst"]), "baidu"
    print(f"[TRANSLATE] baidu response without result: {body}")
    return None


def plain_ascii(value: str, limit: int = 32) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9 +\\-_/]", " ", ascii_text)
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    return ascii_text[:limit]


def find_chrome_exe() -> str | None:
    candidates = [
        os.environ.get("CHROME_EXE", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def valid_deepseek_pair(chinese: str, pinyin: str) -> tuple[str, str] | None:
    chinese = re.sub(r"\s+", "", str(chinese or "")).strip(" ;:,.")
    pinyin = plain_ascii(re.sub(r"\s+", " ", str(pinyin or "")).strip(" ;:,."))
    pinyin_lower = pinyin.lower()
    bad_tokens = (
        "<",
        ">",
        "English",
        "Chinese",
        "actual Chinese",
        "actual pinyin",
        "Chinese translation",
        "pinyin with spaces",
    )
    if not chinese or any(token in chinese for token in bad_tokens):
        return None
    if pinyin and any(token.lower() in pinyin_lower for token in bad_tokens):
        return None
    if not pinyin or re.match(r"^(py|cn|pinyin)\b", pinyin_lower):
        return None
    if not re.search(r"[\u4e00-\u9fff]", chinese):
        return None
    return chinese[:32], pinyin[:32]


def parse_deepseek_translation(text: str) -> tuple[str, str] | None:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return None

    candidates: list[tuple[str, str]] = []

    for cn_match in re.finditer(r"\bCN\s*[:=]\s*(.+?)(?=(?:[;；]\s*PY\s*[:=]|\s+PY\s*[:=]|$))", cleaned, re.I):
        tail = cleaned[cn_match.start() : cn_match.end() + 160]
        py_match = re.search(r"(?:[;；]\s*)?PY\s*[:=]\s*([^;；。,\n]+)", tail, re.I)
        candidates.append((cn_match.group(1), py_match.group(1) if py_match else ""))

    zh_pattern = r"[\u4e00-\u9fff]+"
    py_pattern = r"[A-Za-z][A-Za-z ]{0,60}"
    for match in re.finditer(rf"({zh_pattern})\s*[/,，;；]\s*({py_pattern})", cleaned):
        candidates.append((match.group(1), match.group(2)))

    for chinese, pinyin in reversed(candidates):
        valid = valid_deepseek_pair(chinese, pinyin)
        if valid:
            return valid
    return None


class DeepSeekTranslator:
    def __init__(self, profile_dir: Path, browser_path: str | None = None, login_timeout_s: int = 180) -> None:
        self.profile_dir = profile_dir
        self.browser_path = browser_path or find_chrome_exe()
        self.login_timeout_s = login_timeout_s
        self.driver: Any = None
        self.ready = False
        self.starting = False
        self.lock = threading.Lock()

    def start(self) -> bool:
        if self.driver is not None:
            return True
        if not self.browser_path:
            print("[DEEPSEEK] browser not found")
            return False
        try:
            from selenium import webdriver  # type: ignore
            from selenium.webdriver.chrome.options import Options  # type: ignore
        except Exception as exc:
            print(f"[DEEPSEEK] selenium unavailable: {exc}")
            return False

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        options = Options()
        options.binary_location = self.browser_path
        options.add_argument(f"--user-data-dir={self.profile_dir}")
        options.add_argument("--profile-directory=Default")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(60)
            self.driver.get(DEEPSEEK_URL)
            print("[DEEPSEEK] browser opened, please login if needed")
            return True
        except Exception as exc:
            print(f"[DEEPSEEK] start failed: {exc}")
            self.driver = None
            return False

    def wait_login(self, timeout_s: int | None = None) -> bool:
        if not self.start() or self.driver is None:
            return False
        timeout_s = self.login_timeout_s if timeout_s is None else timeout_s
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._find_input() is not None:
                print("[DEEPSEEK] chat input detected")
                return True
            time.sleep(2)
        print("[DEEPSEEK] login wait timeout, fallback translators remain enabled")
        return False

    def prime(self) -> bool:
        if self.ready:
            return True
        if not self.wait_login():
            return False
        try:
            self._send_prompt(DEEPSEEK_SYSTEM_PROMPT)
            time.sleep(2)
            self.ready = True
            print("[DEEPSEEK] translator prompt sent")
            return True
        except Exception as exc:
            print(f"[DEEPSEEK] prime failed: {exc}")
            return False

    def prime_background(self) -> None:
        if self.ready or self.starting:
            return

        def worker() -> None:
            self.starting = True
            try:
                self.prime()
            finally:
                self.starting = False

        threading.Thread(target=worker, daemon=True).start()

    def translate(self, word: str) -> tuple[str, str] | None:
        if not word:
            return None
        with self.lock:
            if not self.ready and not self.prime():
                return None
            before = self._last_response_text()
            prompt = (
                f"Translate this English word: {word}\n"
                "Output exactly one line: CN=<actual Chinese characters>; PY=<actual pinyin with spaces>"
            )
            try:
                self._send_prompt(prompt)
                answer = self._wait_response_change(before, 60)
            except Exception as exc:
                print(f"[DEEPSEEK] translate failed: {exc}")
                return None
            parsed = parse_deepseek_translation(answer)
            if parsed:
                print(f"[DEEPSEEK] {word} -> {parsed[0]} / {parsed[1]}")
                return parsed
            print(f"[DEEPSEEK] parse failed: {answer[:120]}")
            return None

    def _find_input(self) -> Any:
        if self.driver is None:
            return None
        selectors = [
            "textarea",
            "div[contenteditable='true']",
            "[role='textbox']",
        ]
        for selector in selectors:
            try:
                elements = self.driver.find_elements("css selector", selector)
            except Exception:
                continue
            for element in elements:
                try:
                    if element.is_displayed() and element.is_enabled():
                        return element
                except Exception:
                    continue
        return None

    def _send_prompt(self, prompt: str) -> None:
        from selenium.webdriver.common.keys import Keys  # type: ignore

        box = self._find_input()
        if box is None:
            raise RuntimeError("DeepSeek input box not found")
        box.click()
        try:
            box.clear()
        except Exception:
            pass
        box.send_keys(prompt)
        box.send_keys(Keys.ENTER)
        time.sleep(0.5)
        if self._last_response_text().strip().endswith(prompt.strip()[:20]):
            box.send_keys(Keys.CONTROL, Keys.ENTER)
            time.sleep(0.5)
        if self._last_response_text().strip().endswith(prompt.strip()[:20]):
            self._click_send_button()

    def _click_send_button(self) -> None:
        if self.driver is None:
            return
        selectors = [
            "button[type='submit']",
            "button[aria-label*='send' i]",
            "button[aria-label*='鍙戦€?]",
            "button[data-testid*='send' i]",
        ]
        for selector in selectors:
            try:
                buttons = self.driver.find_elements("css selector", selector)
            except Exception:
                continue
            for button in reversed(buttons):
                try:
                    if button.is_displayed() and button.is_enabled():
                        button.click()
                        return
                except Exception:
                    continue
        try:
            self.driver.execute_script(
                """
const buttons = Array.from(document.querySelectorAll('button')).filter(b => {
  const r = b.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && !b.disabled;
});
if (buttons.length) buttons[buttons.length - 1].click();
"""
            )
        except Exception:
            return

    def _last_response_text(self) -> str:
        if self.driver is None:
            return ""
        script = r"""
const nodes = Array.from(document.querySelectorAll('main, [role="main"], .markdown, [class*="markdown"], [class*="message"], [class*="content"]'));
const texts = nodes.map(n => (n.innerText || n.textContent || '').trim()).filter(Boolean);
return texts.length ? texts[texts.length - 1] : (document.body.innerText || '');
"""
        try:
            return str(self.driver.execute_script(script) or "")
        except Exception:
            return ""

    def _wait_response_change(self, before: str, timeout_s: int) -> str:
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            current = self._last_response_text()
            if current and current != before:
                if before and current.startswith(before):
                    target = current[len(before) :]
                else:
                    target = current
                last = target
                parsed = parse_deepseek_translation(target)
                if parsed:
                    return target
            time.sleep(1)
        return last or self._last_response_text()


def deepseek_translate(word: str) -> tuple[str, str] | None:
    if not DEEPSEEK_ENABLED or DEEPSEEK_TRANSLATOR is None:
        return None
    result = DEEPSEEK_TRANSLATOR.translate(word)
    if result:
        chinese, pinyin = result
        pinyin = plain_ascii(pinyin)
        if pinyin and not re.match(r"^(py|cn|pinyin)\b", pinyin.lower()):
            ASCII_TRANSLATIONS[word] = pinyin[:32]
        return chinese, "deepseek"
    return None


def start_deepseek_if_enabled(args: argparse.Namespace) -> None:
    global DEEPSEEK_ENABLED, DEEPSEEK_TRANSLATOR

    if not getattr(args, "deepseek", False):
        return

    profile_dir = Path(args.deepseek_profile)
    browser_path = args.deepseek_browser or None
    DEEPSEEK_TRANSLATOR = DeepSeekTranslator(
        profile_dir=profile_dir,
        browser_path=browser_path,
        login_timeout_s=max(30, int(args.deepseek_login_timeout)),
    )
    DEEPSEEK_ENABLED = True
    print("[DEEPSEEK] enabled")
    print(f"[DEEPSEEK] profile: {profile_dir}")
    print("[DEEPSEEK] Chrome will open; login once, then keep this browser window open.")
    DEEPSEEK_TRANSLATOR.prime_background()


def translate_word(word: str) -> tuple[str, str]:
    deepseek = deepseek_translate(word)
    if deepseek:
        return deepseek
    cloud = baidu_translate(word)
    if cloud:
        return cloud
    return local_translate(word)


def ascii_translation(word: str, translation: str) -> str:
    if word in ASCII_TRANSLATIONS:
        cached = plain_ascii(ASCII_TRANSLATIONS[word])
        if cached and not re.match(r"^(py|cn|pinyin)\b", cached.lower()):
            return cached
    ascii_text = plain_ascii(translation)
    if ascii_text:
        return ascii_text[:32]
    return f"{word} no dict"[:32]


def gbk_hex_text(text: str, limit_chars: int = 4) -> str:
    """Return comma separated GB2312/GBK two-byte codes for STM32 font lookup."""
    codes: list[str] = []
    for ch in str(text or ""):
        if not ("\u4e00" <= ch <= "\u9fff"):
            continue
        try:
            raw = ch.encode("gb2312")
        except UnicodeEncodeError:
            try:
                raw = ch.encode("gbk")
            except UnicodeEncodeError:
                continue
        if len(raw) == 2 and raw[0] >= 0xA1 and raw[1] >= 0xA1:
            codes.append(f"{raw[0]:02X}{raw[1]:02X}")
        if len(codes) >= limit_chars:
            break
    return ",".join(codes)


def zh_code_translation(word: str, translation: str = "") -> str:
    dynamic = gbk_hex_text(translation)
    if dynamic:
        return dynamic
    return ZH_CODE_TRANSLATIONS.get(word, "")


def speak_async(text: str) -> str:
    if not TTS_ENABLED:
        return "disabled"
    if not text:
        return "empty"

    def worker() -> None:
        pythoncom = None
        try:
            try:
                import pythoncom  # type: ignore

                pythoncom.CoInitialize()
            except Exception:
                pythoncom = None

            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            print(f"[TTS] skipped: {exc}")
        finally:
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    threading.Thread(target=worker, daemon=True).start()
    return "queued"


def speak_characters_async(text: str) -> str:
    clean = "".join(ch for ch in str(text or "") if ch.strip())
    if not clean:
        return "empty"
    return speak_async(" ".join(clean))


def load_pc_cnn() -> tuple[Any, Any] | tuple[None, None]:
    with PC_CNN_LOCK:
        if PC_CNN_CACHE["loaded"]:
            return PC_CNN_CACHE["model"], PC_CNN_CACHE["device"]
        if PC_CNN_CACHE["error"]:
            return None, None
        try:
            import torch  # type: ignore

            if str(PC_CNN_CODE_DIR) not in sys.path:
                sys.path.insert(0, str(PC_CNN_CODE_DIR))
            from emnist_common import load_checkpoint  # type: ignore

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_checkpoint(PC_CNN_MODEL_PATH, device)
            PC_CNN_CACHE.update({"model": model, "device": device, "loaded": True, "error": None})
            print(f"[PC-CNN] loaded {PC_CNN_MODEL_PATH} on {device}")
            return model, device
        except Exception as exc:
            PC_CNN_CACHE["error"] = str(exc)
            print(f"[PC-CNN] load failed: {exc}")
            return None, None


def decode_hex_bitmap(bits_hex: str, width: int, height: int) -> Any:
    import numpy as np  # type: ignore

    raw = bytes.fromhex(str(bits_hex or ""))
    total = width * height
    out = np.zeros(total, dtype=np.uint8)
    index = 0
    for value in raw:
        for bit in range(7, -1, -1):
            if index >= total:
                break
            out[index] = 255 if (value & (1 << bit)) else 0
            index += 1
    return out.reshape((height, width))


def segment_bitmap(bitmap: Any) -> list[tuple[int, int, int, int]]:
    height, width = bitmap.shape
    segments: list[tuple[int, int, int, int]] = []
    in_segment = False
    start_x = 0

    for x in range(width):
        has_column = bool((bitmap[:, x] > 0).any())
        if has_column and not in_segment:
            start_x = x
            in_segment = True
        elif not has_column and in_segment:
            end_x = x - 1
            if end_x - start_x + 1 >= PC_CNN_MIN_SEGMENT_W:
                ys = [int(y) for y in range(height) if bool((bitmap[y, start_x : end_x + 1] > 0).any())]
                if ys:
                    segments.append((start_x, end_x, min(ys), max(ys)))
            in_segment = False

    if in_segment:
        end_x = width - 1
        if end_x - start_x + 1 >= PC_CNN_MIN_SEGMENT_W:
            ys = [int(y) for y in range(height) if bool((bitmap[y, start_x : end_x + 1] > 0).any())]
            if ys:
                segments.append((start_x, end_x, min(ys), max(ys)))

    return segments[:PC_CNN_MAX_SEGMENTS]


def normalize_segment(bitmap: Any, box: tuple[int, int, int, int]) -> Any:
    import numpy as np  # type: ignore

    x0, x1, y0, y1 = box
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    long_side = max(width, height)
    scaled_w = max(1, min(PC_CNN_NORMAL_SIZE, int(round(width * PC_CNN_NORMAL_SIZE / long_side))))
    scaled_h = max(1, min(PC_CNN_NORMAL_SIZE, int(round(height * PC_CNN_NORMAL_SIZE / long_side))))
    offset_x = (PC_CNN_GRID_SIZE - scaled_w) // 2
    offset_y = (PC_CNN_GRID_SIZE - scaled_h) // 2
    out = np.zeros((PC_CNN_GRID_SIZE, PC_CNN_GRID_SIZE), dtype=np.float32)

    for sy in range(y0, y1 + 1):
        for sx in range(x0, x1 + 1):
            if bitmap[sy, sx] > 0:
                tx = min(PC_CNN_GRID_SIZE - 1, offset_x + ((sx - x0) * scaled_w // width))
                ty = min(PC_CNN_GRID_SIZE - 1, offset_y + ((sy - y0) * scaled_h // height))
                out[ty, tx] = 1.0

    coords = np.argwhere(out > 0)
    if coords.size:
        cy, cx = coords.mean(axis=0)
        shift_x = int(round((PC_CNN_GRID_SIZE // 2) - cx))
        shift_y = int(round((PC_CNN_GRID_SIZE // 2) - cy))
        shifted = np.zeros_like(out)
        for y, x in coords:
            ny = int(y) + shift_y
            nx = int(x) + shift_x
            if 0 <= ny < PC_CNN_GRID_SIZE and 0 <= nx < PC_CNN_GRID_SIZE:
                shifted[ny, nx] = out[y, x]
        out = shifted

    return out


def pc_cnn_predict_bitmap(item: dict[str, Any]) -> dict[str, Any]:
    import torch  # type: ignore

    width = int(item.get("w") or 0)
    height = int(item.get("h") or 0)
    bits = str(item.get("bits") or "")
    if width <= 0 or height <= 0 or not bits:
        raise ValueError("missing bitmap")

    model, device = load_pc_cnn()
    if model is None or device is None:
        raise RuntimeError(PC_CNN_CACHE.get("error") or "pc cnn unavailable")
    if str(PC_CNN_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(PC_CNN_CODE_DIR))
    bitmap = decode_hex_bitmap(bits, width, height)
    boxes = segment_bitmap(bitmap)
    if not boxes:
        boxes = [(0, width - 1, 0, height - 1)]

    chars: list[str] = []
    confidences: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for box in boxes:
            image = normalize_segment(bitmap, box)
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            classes = getattr(model, "emnist_classes", None)
            pred_id = int(pred.item())
            if classes is not None and 0 <= pred_id < len(classes):
                chars.append(str(classes[pred_id]))
            else:
                from emnist_common import class_text  # type: ignore

                chars.append(str(class_text(pred_id)))
            confidences.append(float(conf.item()))
    infer_ms = (time.perf_counter() - started) * 1000.0
    return {
        "text": "".join(chars),
        "segments": len(boxes),
        "conf": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "infer_ms": round(infer_ms, 2),
    }

def build_pc_cnn_response(item: dict[str, Any]) -> dict[str, Any] | None:
    mode = str(item.get("mode") or "").lower()
    if mode not in PC_CNN_MODES:
        return None
    try:
        result = pc_cnn_predict_bitmap(item)
        text = result["text"]
        tts = speak_characters_async(text)
        return {
            "result": text,
            "mode": "pc_cnn_result",
            "model": "PC-CNN",
            "segments": result["segments"],
            "conf": result["conf"],
            "infer_ms": result["infer_ms"],
            "tts": tts,
        }
    except Exception as exc:
        print(f"[PC-CNN] infer failed: {exc}")
        return {
            "result": "PC CNN FAIL",
            "mode": "pc_cnn_result",
            "model": "PC-CNN",
            "tts": "skipped",
        }

def build_translate_response(item: dict[str, Any]) -> dict[str, Any] | None:
    mode = str(item.get("mode") or "").lower()
    if mode not in TRANSLATE_MODES:
        return None

    raw_word = item.get("word") or item.get("text") or item.get("raw") or ""
    word = clean_word(raw_word)
    translation, engine = translate_word(word)
    ascii_text = ascii_translation(word, translation)
    tts = speak_async(translation)
    return {
        "source": "pc",
        "mode": f"{mode}_result",
        "ok": True,
        "word": word,
        "zh_code": zh_code_translation(word, translation),
        "ascii": ascii_text,
        "text": translation,
        "translation": translation,
        "engine": engine,
        "tts": tts,
    }


def build_metrics() -> dict[str, Any]:
    with STATE_LOCK:
        history = list(STATE["history"])
        latest = dict(STATE["latest"] or {})
        tcp_clients = STATE["tcp_clients"]
        updated_at = STATE["updated_at"]
        started_at = STATE["started_at"]

    model_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    model_latency_sum: dict[str, int] = {}
    model_latency_count: dict[str, int] = {}
    sdtest: list[dict[str, Any]] = []
    latency_points: list[dict[str, Any]] = []
    pc_cnn_results: list[dict[str, Any]] = []
    pc_cnn_ms_sum = 0.0
    pc_cnn_ms_count = 0
    pc_cnn_conf_sum = 0.0
    pc_cnn_conf_count = 0

    for item in history:
        model = str(item.get("model") or "-")
        mode = str(item.get("mode") or "-")
        model_counts[model] = model_counts.get(model, 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        infer_us = item.get("infer_us")
        infer_ms = item.get("infer_ms")
        conf = item.get("conf")
        text = item.get("result") or item.get("text") or item.get("raw") or "-"
        if isinstance(infer_us, int) and infer_us > 0:
            model_latency_sum[model] = model_latency_sum.get(model, 0) + infer_us
            model_latency_count[model] = model_latency_count.get(model, 0) + 1
            latency_points.append(
                {
                    "time": item.get("received_at", "-"),
                    "model": model,
                    "mode": mode,
                    "text": text,
                    "infer_us": infer_us,
                }
            )
        elif isinstance(infer_ms, (int, float)) and infer_ms > 0:
            latency_points.append(
                {
                    "time": item.get("received_at", "-"),
                    "model": model,
                    "mode": mode,
                    "text": text,
                    "infer_us": int(float(infer_ms) * 1000),
                }
            )

        if mode == "sdtest":
            sdtest.append(item)
        if mode == "pc_cnn_result":
            pc_cnn_results.append(item)
            if isinstance(infer_ms, (int, float)) and infer_ms > 0:
                pc_cnn_ms_sum += float(infer_ms)
                pc_cnn_ms_count += 1
            if isinstance(conf, (int, float)) and conf > 0:
                pc_cnn_conf_sum += float(conf)
                pc_cnn_conf_count += 1

    model_latency_avg = {
        model: round(model_latency_sum[model] / model_latency_count[model])
        for model in model_latency_sum
        if model_latency_count.get(model)
    }

    return {
        "latest": latest or None,
        "history": history,
        "tcp_clients": tcp_clients,
        "updated_at": updated_at,
        "started_at": started_at,
        "summary": {
            "total_events": len(history),
            "model_counts": model_counts,
            "mode_counts": mode_counts,
            "model_latency_avg": model_latency_avg,
            "sdtest": sdtest[-8:],
            "latency_points": latency_points[-40:],
            "pc_cnn_results": pc_cnn_results[-12:],
            "pc_cnn_avg_ms": round(pc_cnn_ms_sum / pc_cnn_ms_count, 2) if pc_cnn_ms_count else None,
            "pc_cnn_avg_conf": round(pc_cnn_conf_sum / pc_cnn_conf_count, 4) if pc_cnn_conf_count else None,
        },
    }


def build_stm32_reply(item: dict[str, Any]) -> dict[str, Any]:
    mode = str(item.get("mode") or "")
    if mode == "pc_cnn_result":
        return {
            "result": str(item.get("result") or ""),
            "mode": "pc_cnn_result",
        }
    if mode in {"word_result", "translate_result"}:
        zh_code = str(item.get("zh_code") or "")
        ascii_text = str(item.get("ascii") or item.get("pinyin") or "")
        if zh_code:
            result_text = f"ZHCODE:{zh_code[:24]};PY:{ascii_text[:32]}"
        else:
            result_text = ascii_text[:32] or str(item.get("translation") or item.get("text") or "")[:32]
        reply: dict[str, Any] = {
            "result": result_text,
            "mode": mode,
            "word": str(item.get("word") or "")[:16],
        }
        if zh_code:
            reply["zh_code"] = zh_code[:24]
        if ascii_text:
            reply["ascii"] = ascii_text[:32]
            reply["text"] = ascii_text[:32]
        return reply
    return item


def send_json_line(sock: Any, item: dict[str, Any]) -> None:
    payload = json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n"
    sock.sendall(payload.encode("utf-8"))
    print(f"[PC->STM32] {payload.strip()}")


class Stm32TcpHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        with STATE_LOCK:
            STATE["tcp_clients"] += 1
        print(f"[TCP] connected: {peer}")
        buf = b""
        try:
            while True:
                data = self.request.recv(1024)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    try:
                        item = json.loads(text)
                    except json.JSONDecodeError:
                        item = {"source": "stm32", "raw": text}
                    print(f"[STM32] {item}")
                    add_event(item)
                    response = None
                    if isinstance(item, dict):
                        response = build_pc_cnn_response(item)
                        if response is None:
                            response = build_translate_response(item)
                    if response is not None:
                        add_event(response)
                        send_json_line(self.request, build_stm32_reply(response))
        finally:
            with STATE_LOCK:
                STATE["tcp_clients"] = max(0, STATE["tcp_clients"] - 1)
            print(f"[TCP] disconnected: {peer}")


class ThreadedTcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STM32 神经网络手写识别展示平台</title>
  <style>
    :root {
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: #172033;
      background: #edf2f7;
      --panel: #ffffff;
      --line: #d8e1ec;
      --muted: #65728a;
      --ink: #172033;
      --blue: #1f6feb;
      --cyan: #0f8b8d;
      --green: #248a57;
      --red: #c2410c;
      --dark: #192338;
      --amber: #b7791f;
      --violet: #6b46c1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, #f6f9fc 0%, #edf2f7 48%, #e9eff5 100%);
    }
    main { width: min(1240px, calc(100% - 32px)); margin: 0 auto; padding: 24px 0; }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 0.42fr);
      gap: 18px;
      align-items: center;
      padding: 22px 24px;
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(25, 35, 56, 0.98), rgba(29, 57, 92, 0.98));
      color: white;
      box-shadow: 0 14px 36px rgba(23, 32, 51, 0.16);
    }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    .sub { margin: 8px 0 0; color: #c7d4e7; line-height: 1.55; }
    .badge { padding: 8px 12px; border-radius: 6px; background: var(--green); font-weight: 700; white-space: nowrap; }
    .hero-live { border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.09); border-radius: 8px; padding: 16px; }
    .hero-live span { display: block; color: #bfd0e8; font-size: 13px; margin-bottom: 8px; }
    .hero-live strong { display: block; font-size: 42px; line-height: 1; overflow-wrap: anywhere; }
    .hero-live small { color: #d6e3f5; line-height: 1.45; }
    .toolbar { display: flex; gap: 8px; justify-content: flex-end; align-items: center; flex-wrap: wrap; }
    button {
      border: 1px solid #bac7d8;
      background: white;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 700;
    }
    button.active { background: var(--blue); border-color: var(--blue); color: white; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 16px 0; }
    .card { min-height: 112px; padding: 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    .card span { display: block; color: var(--muted); font-size: 13px; margin-bottom: 10px; }
    .card strong { display: block; overflow-wrap: anywhere; font-size: 28px; line-height: 1.2; }
    .card small { color: var(--muted); line-height: 1.5; }
    .wide { grid-column: span 2; }
    .result-card { background: linear-gradient(145deg, #ffffff 0%, #edf6ff 100%); border-color: #b9d1ee; }
    .result-card strong { font-size: 36px; color: var(--blue); }
    .layout { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.9fr); gap: 14px; align-items: start; }
    section { padding: 18px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    h2 { margin: 0 0 12px; font-size: 18px; }
    h3 { margin: 0 0 8px; font-size: 15px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
    .panel { display: none; }
    .panel.active { display: block; }
    .pipeline { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .step { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #f8fafc; min-height: 92px; }
    .step b { display: block; margin-bottom: 8px; }
    .mode-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .mode-card { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fbfdff; }
    .mode-card.active-mode { border-color: #8bb7f0; box-shadow: inset 3px 0 0 var(--blue); background: #f3f8ff; }
    .mode-card p { margin: 0; color: var(--muted); line-height: 1.55; }
    .metric-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }
    .mini-metric { border: 1px solid var(--line); background: #fbfdff; border-radius: 8px; padding: 10px; }
    .mini-metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
    .mini-metric b { font-size: 18px; overflow-wrap: anywhere; }
    .model-diagram { display: grid; gap: 14px; }
    .diagram-row { display: grid; grid-template-columns: 126px minmax(0, 1fr); gap: 12px; align-items: start; }
    .diagram-title { font-weight: 800; color: var(--ink); padding-top: 9px; }
    .nodes { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .node { border: 1px solid #c9d8ea; border-radius: 7px; padding: 8px 10px; background: #f8fbff; min-width: 82px; text-align: center; font-weight: 700; }
    .node small { display: block; margin-top: 3px; color: var(--muted); font-weight: 500; }
    .arrow { color: #8aa1bd; font-weight: 800; align-self: center; }
    .feature-list { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }
    .feature-list li { padding: 10px 12px; border: 1px solid #e1e8f2; border-radius: 8px; background: #fbfdff; line-height: 1.5; }
    .feature-list b { color: var(--ink); }
    .bars { display: grid; gap: 10px; }
    .bar-row { display: grid; grid-template-columns: 72px minmax(0, 1fr) 80px; gap: 10px; align-items: center; }
    .bar { height: 10px; background: #e5edf6; border-radius: 999px; overflow: hidden; }
    .bar i { display: block; height: 100%; background: var(--blue); width: 0%; }
    .diag-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .diag { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfdff; }
    .diag span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .diag b { font-size: 20px; overflow-wrap: anywhere; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 8px; border-radius: 999px; background: #edf6ff; color: #205493; font-weight: 700; font-size: 12px; }
    .pill.green { background: #e9f7ef; color: #19603a; }
    .pill.amber { background: #fff7e6; color: #8a5a10; }
    .pill.violet { background: #f2edff; color: #5835a2; }
    .chart { width: 100%; height: 210px; border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; }
    .empty { color: var(--muted); padding: 18px; text-align: center; border: 1px dashed var(--line); border-radius: 8px; }
    .split { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .progress { height: 10px; border-radius: 999px; background: #e8eef6; overflow: hidden; margin-top: 8px; }
    .progress i { display: block; height: 100%; width: 0%; background: var(--green); }
    .timeline { display: grid; gap: 8px; }
    .event { display: grid; grid-template-columns: 92px 76px minmax(0, 1fr) 90px; gap: 8px; align-items: center; padding: 9px 10px; border: 1px solid #e3ebf5; border-radius: 8px; background: #fbfdff; }
    .event:first-child { border-color: #a9c8f0; background: #f3f8ff; }
    .event b, .event span, .event code { overflow-wrap: anywhere; }
    .status-dot { display: inline-flex; align-items: center; gap: 6px; }
    .status-dot::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 3px rgba(36,138,87,0.13); }
    .status-dot.idle::before { background: #9aa8ba; box-shadow: 0 0 0 3px rgba(101,114,138,0.12); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid #e7edf5; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; }
    code { font-family: Consolas, monospace; }
    @media (max-width: 980px) {
      header { grid-template-columns: 1fr; }
      .toolbar { justify-content: flex-start; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .wide { grid-column: span 2; }
      .layout { grid-template-columns: 1fr; }
      .pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .diag-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .mode-grid, .split, .metric-row { grid-template-columns: 1fr; }
      .diagram-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      main { width: min(100% - 20px, 1240px); }
      .grid { grid-template-columns: 1fr; }
      .wide { grid-column: span 1; }
      .pipeline { grid-template-columns: 1fr; }
      .diag-grid { grid-template-columns: 1fr; }
      .event { grid-template-columns: 1fr; }
      h1 { font-size: 23px; }
      .card strong { font-size: 24px; }
      .hero-live strong { font-size: 34px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>STM32 神经网络手写识别展示平台</h1>
      <p class="sub">触摸屏采集手写轨迹，完成 28×28 归一化、FNN/CNN 推理、TF 卡批量测试，并通过 ESP8266 将结果实时推送到网页。</p>
    </div>
    <div class="hero-live">
      <span>当前识别结果</span>
      <strong id="heroText">-</strong>
      <small id="heroMeta">等待 STM32 数据接入</small>
      <div style="margin-top:12px;"><span class="badge" id="status">等待数据</span></div>
    </div>
  </header>

  <div class="grid">
    <div class="card"><span>模式</span><strong id="mode">-</strong></div>
    <div class="card"><span>模型</span><strong id="model">-</strong></div>
    <div class="card result-card"><span>最新结果</span><strong id="text">-</strong><small id="latestMeta">等待事件</small></div>
    <div class="card"><span>推理时间</span><strong id="infer">-</strong></div>
    <div class="card wide"><span>更新时间</span><strong id="time">-</strong></div>
    <div class="card"><span>TCP 客户端</span><strong id="clients">0</strong></div>
    <div class="card"><span>累计事件</span><strong id="events">0</strong></div>
  </div>

  <div class="layout">
    <section>
      <div class="tabs">
        <button class="tab active" data-tab="overview">项目总览</button>
        <button class="tab" data-tab="modes">功能模式</button>
        <button class="tab" data-tab="models">模型结构</button>
        <button class="tab" data-tab="latency">动态指标</button>
        <button class="tab" data-tab="records">实验记录</button>
      </div>

      <div class="panel active" id="tab-overview">
        <h2>端侧 AI 识别流程</h2>
        <div class="pipeline">
          <div class="step"><b>触摸采集</b><small>STM32 采集轨迹坐标，LCD 显示实时笔迹。</small></div>
          <div class="step"><b>预处理</b><small>去噪、边界裁剪、缩放到 28×28 输入图。</small></div>
          <div class="step"><b>模型推理</b><small>KEY2 在 FNN 和 int8 CNN 之间切换。</small></div>
          <div class="step"><b>批量评估</b><small>TF 卡测试集统计准确率和平均运行时间。</small></div>
          <div class="step"><b>WiFi 上报</b><small>ESP8266 将结果发送到电脑网页展示。</small></div>
        </div>
        <div class="metric-row">
          <div class="mini-metric"><span>最近模式</span><b id="overviewMode">-</b></div>
          <div class="mini-metric"><span>最近模型</span><b id="overviewModel">-</b></div>
          <div class="mini-metric"><span>数据刷新</span><b id="refreshAge">-</b></div>
        </div>
        <h2 style="margin-top:18px;">答辩展示重点</h2>
        <ul class="feature-list">
          <li><b>完整闭环：</b>从触摸屏手写输入、图像预处理、神经网络推理到网页端可视化，形成可演示的边缘 AI 系统。</li>
          <li><b>双模型部署：</b>FNN 负责轻量数字识别，CNN/EMNIST 扩展到数字和大小写字母识别。</li>
          <li><b>实验可量化：</b>TF 卡测试集自动统计准确率和平均运行时间，网页实时展示历史记录与耗时趋势。</li>
          <li><b>边缘协同：</b>STM32 负责采集和端侧推理，电脑端可作为 WiFi 数据看板和 PC-CNN 对照诊断。</li>
        </ul>
      </div>

      <div class="panel" id="tab-modes">
        <h2>功能模式界面</h2>
        <div class="mode-grid" id="modeCards">
          <div class="mode-card" data-mode-key="single"><h3>Mode 1：单字符识别</h3><p>触摸屏输入单个数字或字母，STM32 端完成预处理和模型推理，LCD 与网页同步显示识别结果。</p></div>
          <div class="mode-card" data-mode-key="sdtest"><h3>Mode 2：TF 卡批量测试</h3><p>读取 TF 卡测试集，自动比对真实标签，统计准确率和平均推理耗时，适合展示模型性能。</p></div>
          <div class="mode-card" data-mode-key="string"><h3>Mode 3：字符串识别</h3><p>连续书写多个字符，端侧按笔画间隔分割并组合输出，用于展示从单字符到字符串的扩展能力。</p></div>
          <div class="mode-card" data-mode-key="word"><h3>Mode 4：单词翻译</h3><p>逐个识别英文字母组成单词，经 WiFi 发送到电脑端翻译，再回传读音和中文显示。</p></div>
          <div class="mode-card" data-mode-key="pc_cnn_result"><h3>Mode 5：PC-CNN 辅助识别</h3><p>STM32 上传采样图像，电脑端使用完整 EMNIST CNN 识别字符串并返回结果，用于展示边缘协同推理。</p></div>
          <div class="mode-card" data-mode-key="boot"><h3>系统状态</h3><p>展示 WiFi、TCP、模型加载和网页接收状态，方便演示前快速确认链路正常。</p></div>
        </div>
        <h2 style="margin-top:18px;">模型/模式统计</h2>
        <div class="split">
          <div><h3>模型调用次数</h3><div class="bars" id="modelBars"></div></div>
          <div><h3>模式事件次数</h3><div class="bars" id="modeBars"></div></div>
        </div>
      </div>

      <div class="panel" id="tab-models">
        <h2>神经网络结构说明</h2>
        <div class="model-diagram">
          <div class="diagram-row">
            <div class="diagram-title">FNN 数字模型</div>
            <div class="nodes">
              <div class="node">28×28<small>灰度输入</small></div><span class="arrow">→</span>
              <div class="node">Flatten<small>784 维</small></div><span class="arrow">→</span>
              <div class="node">FC + ReLU<small>隐藏层</small></div><span class="arrow">→</span>
              <div class="node">FC10<small>0-9</small></div>
            </div>
          </div>
          <div class="diagram-row">
            <div class="diagram-title">STM32 Tiny CNN</div>
            <div class="nodes">
              <div class="node">28×28<small>字符图</small></div><span class="arrow">→</span>
              <div class="node">Conv16<small>3×3</small></div><span class="arrow">→</span>
              <div class="node">Pool<small>14×14</small></div><span class="arrow">→</span>
              <div class="node">Conv32<small>3×3</small></div><span class="arrow">→</span>
              <div class="node">Pool<small>7×7</small></div><span class="arrow">→</span>
              <div class="node">FC62<small>0-9/a-z/A-Z</small></div>
            </div>
          </div>
          <div class="diagram-row">
            <div class="diagram-title">PC-CNN 对照模型</div>
            <div class="nodes">
              <div class="node">轨迹图像<small>WiFi 上传</small></div><span class="arrow">→</span>
              <div class="node">分割归一化<small>多字符</small></div><span class="arrow">→</span>
              <div class="node">EMNISTCNN<small>完整 PyTorch</small></div><span class="arrow">→</span>
              <div class="node">置信度/结果<small>回传 STM32</small></div>
            </div>
          </div>
        </div>
        <div class="diag-grid">
          <div class="diag"><span>输入规格</span><b>28×28</b><small>二值/灰度笔迹图</small></div>
          <div class="diag"><span>端侧输出</span><b>10 / 62 类</b><small>数字、大小写字母</small></div>
          <div class="diag"><span>部署策略</span><b>轻量化</b><small>FNN + Tiny CNN + PC 协同</small></div>
        </div>
      </div>

      <div class="panel" id="tab-latency">
        <h2>动态指标与推理耗时</h2>
        <div class="metric-row">
          <div class="mini-metric"><span>最近耗时</span><b id="latestInferMetric">-</b></div>
          <div class="mini-metric"><span>平均耗时</span><b id="avgInferMetric">-</b></div>
          <div class="mini-metric"><span>累计事件</span><b id="eventMetric">0</b></div>
        </div>
        <canvas class="chart" id="latencyChart" width="760" height="210"></canvas>
        <div id="latencyAvg" style="margin-top:12px;"></div>
        <h2 style="margin-top:18px;">PC-CNN 诊断</h2>
        <div class="diag-grid">
          <div class="diag"><span>平均 PC 推理</span><b id="pcCnnAvgMs">-</b></div>
          <div class="diag"><span>平均置信度</span><b id="pcCnnAvgConf">-</b></div>
          <div class="diag"><span>最近分段数</span><b id="pcCnnSegments">-</b></div>
        </div>
        <div id="pcCnnList" class="actions"></div>
      </div>

      <div class="panel" id="tab-records">
        <h2>最近记录</h2>
        <div class="actions">
          <button id="exportCsvBtn" type="button">导出 CSV</button>
        </div>
        <div class="timeline" id="eventTimeline"></div>
        <h2 style="margin-top:18px;">原始记录表</h2>
        <table>
          <thead><tr><th>时间</th><th>模式</th><th>模型</th><th>结果</th><th>耗时</th><th>原始数据</th></tr></thead>
          <tbody id="history"><tr><td colspan="6">暂无数据</td></tr></tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>TF 卡批量测试</h2>
      <div id="sdtest" class="empty">等待 SD 批量测试结果</div>
      <h2 style="margin-top:18px;">通信链路</h2>
      <div class="diag-grid">
        <div class="diag"><span>链路状态</span><b id="linkStatus" class="status-dot idle">等待数据</b></div>
        <div class="diag"><span>最近模式</span><b id="linkMode">-</b></div>
        <div class="diag"><span>PC-CNN 返回</span><b id="linkResult">-</b></div>
      </div>
      <h2 style="margin-top:18px;">课程功能覆盖</h2>
      <table>
        <tbody>
          <tr><th>基础识别</th><td>触摸采集、轨迹显示、去噪归一化、单字符推理。</td></tr>
          <tr><th>自动化测试</th><td>TF 卡标准集与个人集批量推理，统计准确率和平均运行时间。</td></tr>
          <tr><th>神经网络训练</th><td>MNIST FNN/CNN 与 EMNIST 扩展模型，网页展示结构与结果。</td></tr>
          <tr><th>联网展示</th><td>ESP8266 上传识别结果，电脑端网页动态显示并导出记录。</td></tr>
        </tbody>
      </table>
    </section>
  </div>
</main>
<script>
function esc(value) {
  return String(value ?? '-').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

function modeLabel(mode) {
  const labels = {
    boot: '系统启动',
    single: '单字符',
    digit: '单字符',
    sdtest: 'TF卡测试',
    string: '字符串',
    word: '单词翻译',
    word_result: '翻译结果',
    pc_cnn: 'PC-CNN请求',
    pc_cnn_result: 'PC-CNN结果'
  };
  return labels[mode] || mode || '-';
}

function modeKey(mode) {
  if (!mode) return '';
  if (mode === 'digit') return 'single';
  if (mode === 'word_result') return 'word';
  if (mode === 'pc_cnn') return 'pc_cnn_result';
  return mode;
}

function drawBarsInto(id, counts) {
  const entries = Object.entries(counts).filter(([, v]) => v > 0);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  const bars = entries.map(([name, value]) => {
    const width = Math.round(value * 100 / max);
    return `<div class="bar-row"><b>${esc(modeLabel(name))}</b><div class="bar"><i style="width:${width}%"></i></div><span>${value}</span></div>`;
  });
  document.getElementById(id).innerHTML = bars.length ? bars.join('') : '<div class="empty">暂无统计数据</div>';
}

function drawBars(summary) {
  drawBarsInto('modelBars', summary.model_counts || {});
  drawBarsInto('modeBars', summary.mode_counts || {});
}

function drawLatency(points) {
  const canvas = document.getElementById('latencyChart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fbfdff';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#d8e1ec';
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = 20 + i * 38;
    ctx.beginPath();
    ctx.moveTo(42, y);
    ctx.lineTo(w - 12, y);
    ctx.stroke();
  }
  if (!points.length) {
    ctx.fillStyle = '#65728a';
    ctx.font = '14px Microsoft YaHei, Segoe UI, Arial';
    ctx.fillText('暂无推理耗时数据', 42, 110);
    return;
  }
  const maxY = Math.max(1, ...points.map(p => p.infer_us || 0));
  const minY = Math.min(...points.map(p => p.infer_us || 0));
  const range = Math.max(1, maxY - minY);
  const plotW = w - 58;
  const plotH = h - 46;
  const x0 = 42;
  const y0 = 14;
  const colors = {FNN: '#1f6feb', CNN: '#c2410c', SYS: '#248a57', PC: '#6b46c1'};
  ctx.font = '12px Microsoft YaHei, Segoe UI, Arial';
  ctx.fillStyle = '#65728a';
  ctx.fillText(maxY + ' us', 4, y0 + 8);
  ctx.fillText(minY + ' us', 4, y0 + plotH);
  points.forEach((p, idx) => {
    const x = x0 + (points.length === 1 ? plotW : idx * plotW / (points.length - 1));
    const y = y0 + plotH - ((p.infer_us - minY) * plotH / range);
    ctx.fillStyle = colors[p.model] || '#0f8b8d';
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    if (idx > 0) {
      const prev = points[idx - 1];
      const px = x0 + ((idx - 1) * plotW / (points.length - 1));
      const py = y0 + plotH - ((prev.infer_us - minY) * plotH / range);
      ctx.strokeStyle = '#8aa1bd';
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
  });
}

function renderSdtest(items) {
  const box = document.getElementById('sdtest');
  if (!items || !items.length) {
    box.className = 'empty';
    box.textContent = '等待 SD 批量测试结果';
    return;
  }
  box.className = '';
  box.innerHTML = items.slice().reverse().map(item => {
    const match = String(item.text || '').match(/(\d+)\/(\d+)\s+([\d.]+)%/);
    const acc = match ? Number(match[3]) : null;
    const width = acc == null ? 0 : Math.max(0, Math.min(100, acc));
    return `<div class="mode-card" style="margin-bottom:10px;"><h3>${esc(item.model || '-')} 批量测试</h3><p><strong>${esc(item.text)}</strong></p><div class="progress"><i style="width:${width}%"></i></div><p style="margin-top:8px;">平均耗时：${item.infer_us ? esc(item.infer_us) + ' us' : '-'}　时间：${esc(item.received_at)}</p></div>`;
  }).join('');
}

function renderPcCnn(summary, latest) {
  document.getElementById('pcCnnAvgMs').textContent = summary.pc_cnn_avg_ms ? summary.pc_cnn_avg_ms + ' ms' : '-';
  document.getElementById('pcCnnAvgConf').textContent = summary.pc_cnn_avg_conf ? summary.pc_cnn_avg_conf.toFixed(4) : '-';
  document.getElementById('pcCnnSegments').textContent = latest.segments || '-';
  const list = summary.pc_cnn_results || [];
  document.getElementById('pcCnnList').innerHTML = list.length
    ? list.slice().reverse().map(item => `<span class="pill">${esc(item.result || '-') } | seg ${esc(item.segments || '-')} | ${item.infer_ms ? esc(item.infer_ms) + ' ms' : '-'}</span>`).join('')
    : '<div class="empty">暂无 PC-CNN 返回记录</div>';
}

function renderModeCards(latest) {
  const active = modeKey(latest.mode);
  document.querySelectorAll('.mode-card[data-mode-key]').forEach(card => {
    card.classList.toggle('active-mode', card.dataset.modeKey === active);
  });
}

function renderTimeline(items) {
  const box = document.getElementById('eventTimeline');
  const rows = (items || []).slice().reverse().slice(0, 8).map(item => {
    const text = item.result || item.text || item.raw || '-';
    const infer = item.infer_us ? item.infer_us + ' us' : (item.infer_ms ? item.infer_ms + ' ms' : '-');
    return `<div class="event"><span>${esc(item.received_at || '-')}</span><b>${esc(modeLabel(item.mode))}</b><strong>${esc(text)}</strong><code>${esc(item.model || '-')}/${esc(infer)}</code></div>`;
  });
  box.innerHTML = rows.length ? rows.join('') : '<div class="empty">暂无历史事件</div>';
}

function newestAgeText(updatedAt) {
  if (!updatedAt) return '-';
  const stamp = Date.parse(String(updatedAt).replace(' ', 'T'));
  if (!Number.isFinite(stamp)) return '已刷新';
  const sec = Math.max(0, Math.round((Date.now() - stamp) / 1000));
  if (sec < 3) return '刚刚';
  if (sec < 60) return sec + ' 秒前';
  return Math.round(sec / 60) + ' 分钟前';
}

function exportHistoryCsv(items) {
  const header = ['received_at','mode','model','text','result','infer_us','infer_ms','segments','conf'];
  const rows = [header.join(',')];
  (items || []).forEach(item => {
    const row = header.map(key => {
      const value = String(item[key] ?? '').replace(/"/g, '""');
      return `"${value}"`;
    });
    rows.push(row.join(','));
  });
  const blob = new Blob([rows.join('\n')], {type: 'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'stm32_wifi_history.csv';
  a.click();
  URL.revokeObjectURL(url);
}

async function loadState() {
  const res = await fetch('/api/state?ts=' + Date.now());
  const state = await res.json();
  const latest = state.latest || {};
  const summary = state.summary || {};
  const latestText = latest.result || latest.text || latest.raw || '-';
  const latestInfer = latest.infer_us ? latest.infer_us + ' us' : (latest.infer_ms ? latest.infer_ms + ' ms' : '-');
  document.getElementById('status').textContent = state.latest ? '实时接收中' : '等待数据';
  document.getElementById('mode').textContent = modeLabel(latest.mode);
  document.getElementById('model').textContent = latest.model || '-';
  document.getElementById('text').textContent = latestText;
  document.getElementById('heroText').textContent = latestText;
  document.getElementById('infer').textContent = latestInfer;
  const metaText = latest.mode === 'pc_cnn_result'
    ? `PC推理 ${latest.infer_ms ? latest.infer_ms + ' ms' : '-'} / 置信度 ${latest.conf ?? '-'}`
    : `来源 ${latest.mode || '-'} / 模型 ${latest.model || '-'}`;
  document.getElementById('latestMeta').textContent = metaText;
  document.getElementById('heroMeta').textContent = metaText;
  document.getElementById('time').textContent = state.updated_at || '-';
  document.getElementById('clients').textContent = state.tcp_clients || 0;
  document.getElementById('events').textContent = summary.total_events || 0;
  document.getElementById('overviewMode').textContent = modeLabel(latest.mode);
  document.getElementById('overviewModel').textContent = latest.model || '-';
  document.getElementById('refreshAge').textContent = newestAgeText(state.updated_at);
  document.getElementById('latestInferMetric').textContent = latestInfer;
  document.getElementById('eventMetric').textContent = summary.total_events || 0;
  const avgValues = Object.values(summary.model_latency_avg || {}).filter(v => Number.isFinite(Number(v)));
  const avgAll = avgValues.length ? Math.round(avgValues.reduce((a, b) => Number(a) + Number(b), 0) / avgValues.length) + ' us' : '-';
  document.getElementById('avgInferMetric').textContent = avgAll;
  const linkStatus = document.getElementById('linkStatus');
  linkStatus.textContent = state.latest ? '接收中' : '等待数据';
  linkStatus.classList.toggle('idle', !state.latest);
  document.getElementById('linkMode').textContent = modeLabel(latest.mode);
  document.getElementById('linkResult').textContent = latestText;
  drawBars(summary);
  drawLatency(summary.latency_points || []);
  renderSdtest(summary.sdtest || []);
  renderPcCnn(summary, latest);
  renderModeCards(latest);
  renderTimeline(state.history || []);
  const latencyAvg = Object.entries(summary.model_latency_avg || {}).map(([model, value]) => `${esc(model)} 平均 ${esc(value)} us`);
  document.getElementById('latencyAvg').innerHTML = latencyAvg.length ? latencyAvg.map(x => `<span class="pill green">${x}</span>`).join('') : '<div class="empty">暂无平均耗时</div>';
  const tbody = document.getElementById('history');
  const rows = (state.history || []).slice().reverse().map(item => {
    const raw = JSON.stringify(item);
    const text = item.result || item.text || item.raw || '-';
    const infer = item.infer_us ? item.infer_us + ' us' : (item.infer_ms ? item.infer_ms + ' ms' : '-');
    return `<tr><td>${esc(item.received_at)}</td><td>${esc(item.mode)}</td><td>${esc(item.model)}</td><td><strong>${esc(text)}</strong></td><td>${esc(infer)}</td><td><code>${esc(raw)}</code></td></tr>`;
  });
  tbody.innerHTML = rows.length ? rows.join('') : '<tr><td colspan="6">暂无数据</td></tr>';
  document.getElementById('exportCsvBtn').onclick = () => exportHistoryCsv(state.history || []);
}
setInterval(loadState, 600);
loadState();
</script>
</body>
</html>'''


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            body = json.dumps(build_metrics(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def serve(tcp_host: str, tcp_port: int, web_host: str, web_port: int) -> None:
    tcp = ThreadedTcpServer((tcp_host, tcp_port), Stm32TcpHandler)
    http = ThreadingHTTPServer((web_host, web_port), WebHandler)
    threading.Thread(target=tcp.serve_forever, daemon=True).start()
    threading.Thread(target=load_pc_cnn, daemon=True).start()
    print(f"[TCP] listening on {tcp_host}:{tcp_port} for STM32/ESP8266")
    print(f"[WEB] open http://127.0.0.1:{web_port}/")
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        print("\n[EXIT] stopping")
    finally:
        tcp.shutdown()
        http.shutdown()


def main() -> None:
    global TTS_ENABLED, PC_CNN_MODEL_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp-host", default="0.0.0.0")
    parser.add_argument("--tcp-port", type=int, default=8000)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--no-tts", action="store_true", help="disable optional pyttsx3 speech playback")
    parser.add_argument("--pc-cnn-model", default=str(PC_CNN_MODEL_PATH), help="full precision EMNIST CNN checkpoint")
    parser.add_argument("--deepseek", action="store_true", help="enable DeepSeek browser translator for word mode")
    parser.add_argument("--deepseek-profile", default=str(DEEPSEEK_PROFILE_DIR), help="Chrome user-data directory for DeepSeek login")
    parser.add_argument("--deepseek-browser", default="", help="optional Chrome/Edge executable path")
    parser.add_argument("--deepseek-login-timeout", type=int, default=180, help="seconds to wait for DeepSeek login/input")
    args = parser.parse_args()
    TTS_ENABLED = not args.no_tts
    PC_CNN_MODEL_PATH = Path(args.pc_cnn_model)
    start_deepseek_if_enabled(args)
    serve(args.tcp_host, args.tcp_port, args.web_host, args.web_port)


if __name__ == "__main__":
    main()









