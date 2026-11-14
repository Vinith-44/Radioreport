"""Step 10 Gradio UI (gradio 6.x Blocks API: Tab, Image(type="pil"), Audio(type="filepath"), File, Button.click).
Thin: every action calls src/app/pipeline.Pipeline."""
from __future__ import annotations

import tempfile

from src.app.render import DISCLAIMER, draw_boxes, sentences_html


def build(pipe, cfg: dict):
    import gradio as gr

    ac = cfg["app"]
    langs = ["english"] + list(cfg["multilingual"]["languages"]) + [l_ for l_ in ac.get("extra_languages", [])
                                                                      if l_ not in cfg["multilingual"]["languages"]]
    state = {}

    def on_report(img, ind, prior):
        if img is None:
            return "Upload a chest X-ray first.", None, "", ""
        r = pipe.report(img, ind or "", prior or "")
        state.update(result=r, image=img)
        cases = "\n".join(f"- {c['study_id']} (sim {c['sim']:.2f}, agreement {c['agreement']:.2f}): {c['snippet']}"
                          for c in r["cases"]) or "(none)"
        return sentences_html(r["sentences"]), draw_boxes(img, r["boxes"]), cases, \
            f"generation {r['timing_s']['generate']} s, total {r['timing_s']['total']} s - {r['system']}"

    def on_summary(lang):
        if "result" not in state:
            return "Generate a report first.", ""
        s = pipe.summary(state["result"]["report"], lang)
        state["summary"] = s
        flag = "" if s["faithful"] else (f"FLAGGED: the back-translation lost {s['lost']} / added {s['added']} - "
                                         "do not give this summary to the patient without review.")
        return (s["translation"] or s["summary_en"]), flag or "Back-translation check passed (CheXbert labels match)."

    def on_second(img, text):
        if img is None or not (text or "").strip():
            return "Upload the X-ray and paste the radiologist's report."
        r = pipe.second_reader(img, text)
        if not r["possible_misses"]:
            return f"No possible misses flagged. Findings mentioned: {', '.join(r['mentioned']) or 'none'}."
        return "Possible misses (check the image):\n" + "\n".join(
            f"- {m['finding']} (probe p={m['p']}, threshold {m['tau']})" for m in r["possible_misses"])

    def on_voice(audio, lang, current):
        if not audio:
            return current
        return ((current or "") + "\n" + pipe.transcribe(audio, lang)).strip()

    def on_pdf():
        if "result" not in state:
            return None
        from src.app.pdf import build_pdf

        p = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        return build_pdf(state["result"], draw_boxes(state["image"], state["result"]["boxes"]), p,
                         ac["pdf"]["title"], ac["pdf"]["subtitle"], summary=state.get("summary"),
                         indic_font_ttf=ac["pdf"].get("indic_font_ttf"))

    with gr.Blocks(title="Radioreport") as demo:
        gr.Markdown(f"# Radioreport - chest X-ray report assistant\n**{DISCLAIMER}**")
        with gr.Tab("Report"):
            with gr.Row():
                with gr.Column():
                    img = gr.Image(type="pil", label="Frontal chest X-ray")
                    ind = gr.Textbox(label="Indication")
                    prior = gr.Textbox(label="Prior report (optional)", lines=3)
                    go = gr.Button("Generate + verify", variant="primary")
                with gr.Column():
                    out_html = gr.HTML()
                    out_img = gr.Image(label="Boxes behind the verdicts")
                    timing = gr.Markdown()
            cases = gr.Textbox(label="Supporting cases (other patients, style reference only)", lines=4)
            go.click(on_report, [img, ind, prior], [out_html, out_img, cases, timing])
            pdf_btn = gr.Button("Export PDF")
            pdf_file = gr.File(label="PDF")
            pdf_btn.click(on_pdf, [], [pdf_file])
        with gr.Tab("Patient summary"):
            lang = gr.Dropdown(langs, value=langs[1] if len(langs) > 1 else "english", label="Language")
            sbtn = gr.Button("Summarise")
            stext = gr.Textbox(label="Summary", lines=6)
            sflag = gr.Markdown()
            sbtn.click(on_summary, [lang], [stext, sflag])
        with gr.Tab("Second reader"):
            img2 = gr.Image(type="pil", label="Chest X-ray")
            rep2 = gr.Textbox(label="Radiologist's report", lines=8)
            rbtn = gr.Button("Check for possible misses")
            rout = gr.Textbox(label="Result", lines=6)
            rbtn.click(on_second, [img2, rep2], [rout])
        with gr.Tab("Voice editing"):
            vlang = gr.Dropdown(["english"] + list(cfg["multilingual"]["languages"]), value="english", label="Speech")
            audio = gr.Audio(type="filepath", label="Dictate")
            edit = gr.Textbox(label="Report text", lines=10)
            gr.Button("Transcribe and append").click(on_voice, [audio, vlang, edit], [edit])
    return demo
