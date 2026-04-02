import re
import zipfile
import xml.etree.ElementTree as ET

DOCX_PATH = "thesis-iam-writing.docx"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

with zipfile.ZipFile(DOCX_PATH) as zf:
    root = ET.fromstring(zf.read("word/document.xml"))

paras = []
for p in root.findall(".//w:body/w:p", NS):
    text = "".join((t.text or "") for t in p.findall(".//w:t", NS)).strip()
    if not text:
        continue
    text = re.sub(r"\s+", " ", text).strip()
    paras.append(text)

intro_idx = next((i for i, t in enumerate(paras) if t.upper() == "INTRODUCTION"), None)
next_idx = next((i for i in range((intro_idx or -1) + 1, len(paras)) if paras[i].startswith("1.1 ")), None)

if intro_idx is None:
    print("INTRO_HEADING_NOT_FOUND")
elif next_idx is None:
    print("NEXT_HEADING_NOT_FOUND")
else:
    intro_paras = paras[intro_idx + 1 : next_idx]
    print(f"INTRO_PARAGRAPH_COUNT: {len(intro_paras)}")
    if intro_paras:
        print("LAST_INTRO_PARAGRAPH:")
        print(intro_paras[-1])
