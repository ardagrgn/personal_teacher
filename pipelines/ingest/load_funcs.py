from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Union
import re, json, hashlib, datetime as dt


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class loadedDoc:
    text: str
    metadata: Dict[str, str] 


def _clean_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def load_pdf(path: Union[str, Path]) -> loadedDoc:
    import fitz  # PyMuPDF
    p = Path(path)
    doc = fitz.open(p.as_posix())
    pages = [page.get_text("text") for page in doc]
    doc.close()
    text = _clean_text("\n\n".join(pages))
    return loadedDoc(
        text=text,
        metadata={
            "source": "pdf",
            "path": str(p.resolve()),
            "title": p.stem,
            "hash": _sha256(text),
            "loaded_at": _now_iso(),
        },
    )


def load_epub(path: Union[str, Path]) -> loadedDoc:
    from ebooklib import epub
    from bs4 import BeautifulSoup

    # Changes string path to Path object 
    p = Path(path)

    #Reads book from given path, returns an EpubBook object
    book = epub.read_epub(p.as_posix())
    title = ""

    # First try to get title from metadata (DC= Dublin Core) 
    try:
        md = book.get_metadata("DC", "title")
        # if there is title returns it othwerwise uses filename without extension 
        title = (md[0][0] if md else "") or p.stem

    except Exception:

        title = p.stem

    pieces: List[str] = []

    # Iterate through all items in the epub book
    for item in book.get_items():
        # 9 == DOCUMENT (xhtml)
        #We expcet item to have get_type and get_content methods
        if getattr(item, "get_type", lambda: None)() == 9:
            
            #Gets text content from xhtml document
            soup = BeautifulSoup(item.get_content(), "lxml",)
            
            # Append separated text in pieces text
            pieces.append(soup.get_text("\n"))
    text = _clean_text("\n\n".join(pieces))
    return loadedDoc(
        text=text,
        metadata={
            "source": "epub",
            "path": str(p.resolve()),
            "title": title,
            "hash": _sha256(text),
            "loaded_at": _now_iso(),
        },
    )


def load_youtube(
    url_or_id: str,
    prefer_langs: Optional[List[str]] = None,
) -> loadedDoc:
    """
    Gets official captions if possible (fastest, cheapest).
    If not found and asr_fallback=True, downloads audio and transcribes with Whisper.
    """
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

    def _extract_id(s: str) -> str:
        import re
        m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", s)
        return m.group(1) if m else s.strip()

    yt_id = _extract_id(url_or_id)
    url = f"https://www.youtube.com/watch?v={yt_id}"
    prefer_langs = ["en"]

    # Try transcripts
    try:
        transcripts = YouTubeTranscriptApi().list(yt_id)
        print(transcripts)
        # Exact preferred languages
        for lang in prefer_langs:
            try:
                t = transcripts.find_transcript([lang]).fetch()
                text = _clean_text(" ".join(getattr(seg,"text","") for seg in t.snippets))
                
                return loadedDoc(
                    text=text,
                    metadata={
                        "source": "youtube",
                        "url": url,
                        "lang": lang,
                        "mode": "captions",
                        "hash": _sha256(text),
                        "loaded_at": _now_iso(),
                    },
                )
            except Exception as e:
                print(f"Error fetching transcript for: {e}")
                pass

        # Any available transcript
        for tr in transcripts:
            try:
                t = tr.fetch()
                text = _clean_text(" ".join(getattr(seg,"text","") for seg in t.snippets))
                return loadedDoc(
                    text=text,
                    metadata={
                        "source": "youtube",
                        "url": url,
                        "lang": getattr(tr, "language_code", "unknown"),
                        "mode": "captions_any",
                        "hash": _sha256(text),
                        "loaded_at": _now_iso(),
                    },
                )
            except Exception:
                pass

        # If we reach here, no usable transcript found
        #raise NoTranscriptFound(yt_id)
        return transcripts

    except (TranscriptsDisabled, NoTranscriptFound):
        text = ""
        return loadedDoc(
            text=text,
            metadata={
                "source": "youtube",
                "url": url,
                "lang": "asr",
                "mode": "whisper",
                "hash": _sha256(text),
                "loaded_at": _now_iso(),
            },
        )
    


def load_website(url: str, timeout: int = 20) -> loadedDoc:
    """
    Extracts main article text from a web page.
    Primary: trafilatura. Fallback: readability + BeautifulSoup.
    """
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise RuntimeError("trafilatura.fetch_url returned None")
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not text:
            raise RuntimeError("trafilatura.extract returned None")
        meta = trafilatura.extract_metadata(downloaded)
        #title = (meta.title if meta and getattr(meta, "title", None) else "") or ""
        text = _clean_text(text)
        return loadedDoc(
            text=text,
            metadata={
                "source": "website",
                "url": url,
                "title": "",
                "hash": _sha256(text),
                "loaded_at": _now_iso(),
            },
        )
    except Exception:
        import requests
        from bs4 import BeautifulSoup
        
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        
        html = resp.text
        #title = ""
        soup = BeautifulSoup(html, "lxml")
        text = _clean_text(soup.get_text("\n"))
        return loadedDoc(
            text=text,
            metadata={
                "source": "website",
                "url": url,
                "title": "",
                "hash": _sha256(text),
                "loaded_at": _now_iso(),
            },
        )
    

def load_txt(path: Union[str, Path], encoding: Optional[str] = None) -> loadedDoc:
    p = Path(path)
    content: str
    if encoding:
        content = p.read_text(encoding=encoding, errors="ignore")
    else:
        # Try charset-normalizer if available, else fall back to utf-8
        try:
            from charset_normalizer import from_path
            result = from_path(p.as_posix())
            best = result.best()
            content = best.output() if best else p.read_text(encoding="utf-8", errors="ignore")
            print(content)
        except Exception:
            print("expection started")
            content = p.read_text(encoding="utf-8", errors="ignore")
    text = _clean_text(content)
    return loadedDoc(
        text=text,
        metadata={
            "source": "txt",
            "path": str(p.resolve()),
            "title": p.stem,
            "hash": _sha256(text),
            "loaded_at": _now_iso(),
        },
    )


def load_path(path: Union[str, Path]) -> loadedDoc:
    """
    Auto-detect by file extension: .pdf, .epub, .txt
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(p)
    if ext == ".epub":
        return load_epub(p)
    if ext in {".txt", ".md", ".rst", ".log"}:
        return load_txt(p,"utf-8")
    raise ValueError(f"Unsupported file extension for {p.name}")


def load_many(
    urls: Optional[Iterable[str]] = None,
    files: Optional[Iterable[Union[str, Path]]] = None,
    dedupe_by_hash: bool = True,
) -> List[loadedDoc]:
    """
    load many URLs and/or files; returns a list of loadedDoc.
    Dedupe identical text payloads by SHA256 if desired.
    """
    urls = urls or []
    files = files or []
    docs: List[loadedDoc] = []

    # URLs
    for u in urls:
        if u.startswith("http"):
            docs.append(load_website(u))
        else:
            # Treat as YouTube ID if length 11 or youtube link
            if ("youtube.com" in u) or ("youtu.be" in u) or (len(u.strip()) == 11):
                docs.append(load_youtube(u))
            else:
                raise ValueError(f"Unknown URL/ID format: {u}")

    # Files
    for f in files:
        docs.append(load_path(f))

    if dedupe_by_hash:
        seen = set()
        unique: List[loadedDoc] = []
        for d in docs:
            h = d.metadata.get("hash", _sha256(d.text))
            if h in seen:
                continue
            seen.add(h)
            unique.append(d)
        docs = unique

    return docs