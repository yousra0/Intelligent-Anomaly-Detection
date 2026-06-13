"""
app/services/report_gen_docx.py
Génère le rapport Word en remplissant le template exemple_rapport.docx.
Placeholders {{...}}, graphiques matplotlib, tableau et sections dynamiques.
"""
from __future__ import annotations

import io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "exemple_rapport.docx"

# ─── Couleurs PwC ─────────────────────────────────────────────────────────────
_C_ORANGE = "#D04A02"
_C_DARK   = "#293854"
_C_RED    = "#C00000"
_C_WARN   = "#E66E00"
_C_GREEN  = "#008246"
_C_LGRAY  = "#F8F8F8"
_C_MGRAY  = "#B0B0B0"

RGB_ORANGE = RGBColor(208, 74,  2)
RGB_DARK   = RGBColor(41,  56,  84)
RGB_RED    = RGBColor(192, 0,   0)
RGB_WARN   = RGBColor(230, 110, 0)
RGB_GREEN  = RGBColor(0,   130, 70)
RGB_WHITE  = RGBColor(255, 255, 255)
RGB_GRAY   = RGBColor(80,  80,  80)

FEATURE_FR = {
    "balance_diff_orig": "Ecart de solde (emetteur)",
    "log_amount":        "Montant (echelle log)",
    "amount":            "Montant de la transaction",
    "hour":              "Heure de la transaction",
    "dest_zero_balance": "Solde nul chez le destinataire",
    "type_CASH_OUT":     "Type : Retrait especes",
    "type_TRANSFER":     "Type : Virement bancaire",
    "type_CASH_IN":      "Type : Depot especes",
    "type_PAYMENT":      "Type : Paiement",
    "type_DEBIT":        "Type : Debit automatique",
}

FEATURE_EXPLAIN = {
    "balance_diff_orig": (
        "Le solde du compte emetteur a change d'une facon qui ne correspond pas "
        "aux operations declarees. Indicateur tres fort de fraude potentielle."
    ),
    "log_amount": (
        "Le montant est inhabituellement eleve par rapport aux transactions similaires."
    ),
    "hour": (
        "Transaction effectuee a une heure atypique (nuit / tres tot le matin), "
        "periode connue pour concentrer les activites frauduleuses."
    ),
    "dest_zero_balance": (
        "Compte destinataire avec solde nul apres reception : caracteristique "
        "des comptes 'mule' utilises pour blanchiment."
    ),
    "type_CASH_OUT": "Les retraits especes sont la methode la plus frequente dans les fraudes.",
    "type_TRANSFER": "Virements de grande valeur vers comptes inconnus : haut risque.",
}

RISK_LABEL = {
    "CRITIQUE": "A investiguer immediatement",
    "ELEVE":    "A surveiller prioritairement",
    "FAIBLE":   "Aucun probleme detecte",
}

GLOSSAIRE = [
    ("Intelligence artificielle (IA)",
     "Programme informatique capable d'apprendre a partir de donnees historiques "
     "pour detecter des comportements anormaux, sans etre explicitement programme."),
    ("XGBoost",
     "Algorithme d'apprentissage automatique qui analyse de nombreux indicateurs "
     "simultanement pour attribuer un score de risque (0 = normal, 1 = fraude certaine)."),
    ("AutoEncoder",
     "Reseau de neurones qui apprend a 'reconstruire' des transactions normales. "
     "Quand il echoue, cela signale une anomalie."),
    ("SHAP - Facteurs declencheurs",
     "Technique qui explique pourquoi l'IA a signale une transaction en identifiant "
     "les indicateurs les plus contributifs a l'alerte."),
    ("Indice de risque (0 a 1)",
     "Score attribue par l'IA. Plus proche de 1 = plus suspect. "
     "Un seuil calibre sur donnees historiques determine l'alerte."),
    ("Solde nul destinataire",
     "Compte recevant l'argent avec solde nul avant la transaction. "
     "Caracteristique souvent associee aux comptes utilises pour blanchiment."),
    ("Ecart de solde",
     "Difference entre le montant debite du compte source et la variation de solde. "
     "Un ecart important peut indiquer une manipulation des donnees."),
]


# ─── Helpers matplotlib ───────────────────────────────────────────────────────

def _to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def _chart_donut(n_tx: int, n_fraud: int) -> io.BytesIO:
    n_ok   = max(n_tx - n_fraud, 0)
    sizes  = [n_ok, n_fraud] if n_fraud > 0 else [1, 0]
    labels = [f"Normales\n{n_ok:,}", f"Suspectes\n{n_fraud:,}"]
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    fig.patch.set_facecolor("white")
    _, _, autotexts = ax.pie(
        sizes, labels=labels, colors=[_C_DARK, _C_ORANGE],
        autopct="%1.1f%%", startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=8),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8)
    ax.set_title("Repartition des transactions", fontsize=9, fontweight="bold",
                 color=_C_DARK, pad=8)
    plt.tight_layout(pad=0.3)
    return _to_buf(fig)


def _chart_risk_bars(transactions: list[dict]) -> io.BytesIO:
    counts = Counter(t.get("risk_level", "FAIBLE") for t in transactions)
    cats   = ["FAIBLE", "ELEVE", "CRITIQUE"]
    labels = ["Normale", "A surveiller", "Critique"]
    vals   = [counts.get(c, 0) for c in cats]
    hues   = [_C_GREEN, _C_WARN, _C_RED]
    max_v  = max(vals) if max(vals) > 0 else 1

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_C_LGRAY)
    bars = ax.bar(labels, vals, color=hues, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_v * 0.03,
                    str(v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=_C_DARK)
    ax.set_title("Niveau de risque detecte", fontsize=9, fontweight="bold",
                 color=_C_DARK, pad=8)
    ax.set_ylabel("Transactions", fontsize=7, color="#666666")
    ax.set_ylim(0, max_v * 1.25 + 1)
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.3)
    return _to_buf(fig)


def _chart_amount_type(transactions: list[dict]) -> io.BytesIO:
    sums: dict[str, float] = defaultdict(float)
    for t in transactions:
        if t.get("risk_level") in ("CRITIQUE", "ELEVE") or t.get("is_fraud_predicted"):
            sums[t.get("type", "Autre")] += t.get("amount", 0)
    if not sums:
        for t in transactions:
            sums[t.get("type", "Autre")] += t.get("amount", 0)
    items = sorted(sums.items(), key=lambda x: -x[1])[:6] or [("Aucune donnee", 0)]
    types = [i[0] for i in items]
    amts  = [i[1] / 1_000 for i in items]
    pal   = [_C_ORANGE, _C_DARK, _C_RED, _C_WARN, _C_GREEN, _C_MGRAY]
    max_a = max(amts) if max(amts) > 0 else 1

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_C_LGRAY)
    ax.barh(types[::-1], amts[::-1], color=pal[:len(types)][::-1],
            edgecolor="white", linewidth=1.2)
    for bar, v in zip(ax.patches, amts[::-1]):
        ax.text(bar.get_width() + max_a * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f}k TND", va="center", ha="left",
                fontsize=7.5, color=_C_DARK)
    ax.set_title("Exposition financiere par type d'operation (kTND)",
                 fontsize=9, fontweight="bold", color=_C_DARK, pad=8)
    ax.set_xlabel("Milliers TND", fontsize=7, color="#666666")
    ax.set_xlim(0, max_a * 1.4)
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.3)
    return _to_buf(fig)


def _chart_shap(explain_results: list[dict]) -> io.BytesIO | None:
    totals: dict[str, float] = defaultdict(float)
    cnts:   dict[str, int]   = defaultdict(int)
    for e in explain_results:
        for f, v in e.get("shap_values", {}).items():
            totals[f] += abs(float(v))
            cnts[f]   += 1
    if not totals:
        return None
    mean_abs = {f: totals[f] / cnts[f] for f in totals}
    items  = sorted(mean_abs.items(), key=lambda x: -x[1])[:8]
    labels = [FEATURE_FR.get(f, f) for f, _ in items]
    vals   = [v for _, v in items]
    thresh = vals[0] * 0.35 if vals else 0
    hues   = [_C_ORANGE if v >= thresh else _C_DARK for v in vals]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_C_LGRAY)
    ax.barh(labels[::-1], vals[::-1], color=hues[::-1], edgecolor="white", linewidth=1)
    ax.set_title("Facteurs les plus declencheurs d'alertes (importance SHAP)",
                 fontsize=9, fontweight="bold", color=_C_DARK, pad=8)
    ax.set_xlabel("Importance moyenne", fontsize=7, color="#666666")
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.3)
    return _to_buf(fig)


def _fmt_amount(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:,.0f}"


# ─── Helpers docx ─────────────────────────────────────────────────────────────

def _para_text(para) -> str:
    return "".join(r.text or "" for r in para.runs)


def _replace_in_para(para, placeholder: str, new_text: str) -> bool:
    """Replace placeholder handling run-split (Word splits {{...}} across runs)."""
    full = _para_text(para)
    if placeholder not in full:
        return False
    new_full = full.replace(placeholder, new_text)
    if para.runs:
        para.runs[0].text = new_full
        for run in para.runs[1:]:
            run.text = ""
    else:
        para.add_run(new_full)
    return True


_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _replace_all(doc: Document, mapping: dict[str, str]) -> None:
    """Replace all placeholders everywhere: VML textboxes, DrawingML anchors,
    plain paragraph runs, run-split runs, and table cells."""
    # Pass 1: every <w:t> in the entire body (handles VML v:textbox and wp:anchor)
    for t in doc.element.body.iter(f"{{{_NS_W}}}t"):
        if t.text:
            for ph, val in mapping.items():
                if ph in t.text:
                    t.text = t.text.replace(ph, val)

    # Pass 2: run-split placeholders in doc paragraphs (Word splits {{...}} across runs)
    for para in doc.paragraphs:
        for ph, val in mapping.items():
            _replace_in_para(para, ph, val)

    # Pass 3: run-split placeholders inside table cells
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for ph, val in mapping.items():
                        _replace_in_para(para, ph, val)


def _find_para(doc: Document, substr: str):
    for para in doc.paragraphs:
        if substr in para.text:
            return para
    return None


def _insert_chart_after_label(doc: Document, label_substr: str,
                               img_buf: io.BytesIO, width: float = 5.5) -> None:
    """Insert chart image into the first empty paragraph after the label paragraph."""
    label_para = _find_para(doc, label_substr)
    if label_para is None:
        return
    paras = doc.paragraphs
    try:
        idx = paras.index(label_para)
    except ValueError:
        return

    empty_paras: list = []
    for para in paras[idx + 1: idx + 7]:
        if not para.text.strip():
            empty_paras.append(para)
        else:
            break

    if not empty_paras:
        run = label_para.add_run()
        run.add_picture(img_buf, width=Inches(width))
        return

    # Add image to first empty paragraph
    run = empty_paras[0].add_run()
    run.add_picture(img_buf, width=Inches(width))

    # Remove extra empty paragraphs after
    for para in empty_paras[1:]:
        p = para._element
        p.getparent().remove(p)


def _clear_empty_paras_after(doc: Document, ref_para, count: int = 9) -> None:
    paras = doc.paragraphs
    try:
        idx = paras.index(ref_para)
    except ValueError:
        return
    to_remove = []
    for para in paras[idx + 1: idx + 1 + count]:
        if not para.text.strip():
            to_remove.append(para._element)
        else:
            break
    for elem in to_remove:
        elem.getparent().remove(elem)


def _set_cell_bg(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:fill"),  hex_color)
    shd.set(qn("w:color"), hex_color)
    shd.set(qn("w:val"),   "clear")
    tcPr.append(shd)


def _cell_text(cell, text: str, bold: bool = False,
               color: RGBColor = None, size: int = 9,
               align: str = "center") -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    para.alignment = {"center": 1, "left": 0, "right": 2}.get(align, 0)
    run  = para.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color


def _add_para_after(ref_elem, text: str = "", bold: bool = False,
                    italic: bool = False, size: int = 10,
                    color: RGBColor = None, indent: float = 0.0) -> OxmlElement:
    """Insert a new <w:p> element right after ref_elem. Returns the new element."""
    new_p   = OxmlElement("w:p")
    new_pPr = OxmlElement("w:pPr")

    if indent > 0:
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), str(int(indent * 914400 / 2.54 / 10)))  # cm to twips
        new_pPr.append(ind)

    new_p.append(new_pPr)

    if text:
        new_r   = OxmlElement("w:r")
        new_rPr = OxmlElement("w:rPr")

        if bold:
            b = OxmlElement("w:b")
            new_rPr.append(b)
        if italic:
            i = OxmlElement("w:i")
            new_rPr.append(i)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(size * 2))
        new_rPr.append(sz)
        if color:
            clr = OxmlElement("w:color")
            clr.set(qn("w:val"), str(color))   # RGBColor.__str__ -> "RRGGBB"
            new_rPr.append(clr)

        new_r.append(new_rPr)
        new_t = OxmlElement("w:t")
        new_t.text = text
        new_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        new_r.append(new_t)
        new_p.append(new_r)

    ref_elem.addnext(new_p)
    return new_p


def _add_table_after_para(doc: Document, ref_para,
                           headers: list[str], rows: list[list[str]],
                           col_widths: list[float] | None = None) -> None:
    """Add a styled table immediately after ref_para."""
    n_cols = len(headers)
    table  = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.style = "TableGrid"

    # Header row
    hdr_row = table.rows[0]
    for ci, (h, cell) in enumerate(zip(headers, hdr_row.cells)):
        _set_cell_bg(cell, "293854")
        _cell_text(cell, h, bold=True, color=RGB_WHITE, size=8)

    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg  = "F8F8F8" if ri % 2 == 0 else "FFFFFF"
        for ci, (val, cell) in enumerate(zip(row_data, row.cells)):
            _set_cell_bg(cell, bg)
            # Last column = status (colored)
            if ci == n_cols - 1:
                col = RGB_RED if "ALERTE" in val else (RGB_WARN if "ATTENTION" in val else RGB_GREEN)
                _cell_text(cell, val, bold=True, color=col, size=8)
            else:
                _cell_text(cell, val, size=8)

    # Set column widths
    if col_widths:
        for ci, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[ci].width = Inches(w)

    # Move table to just after ref_para
    ref_para._element.addnext(table._tbl)


# ─── Générateur principal ─────────────────────────────────────────────────────

def generate_pwc_docx_report(
    predict_result: dict,
    explain_results: list[dict] | None = None,
) -> bytes:
    now          = datetime.now()
    date_str     = now.strftime("%d/%m/%Y a %H:%M")

    n_tx         = predict_result.get("n_transactions", 0)
    n_fraud      = predict_result.get("n_fraud", 0)
    rate         = predict_result.get("fraud_rate_pct", 0.0)
    amount_risk  = predict_result.get("amount_at_risk", 0.0)
    threshold    = predict_result.get("threshold", 0.355)
    transactions = predict_result.get("transactions", [])

    risk_counts  = Counter(t.get("risk_level", "FAIBLE") for t in transactions)
    n_critique   = risk_counts.get("CRITIQUE", 0)
    n_eleve      = risk_counts.get("ELEVE",    0)
    n_faible     = risk_counts.get("FAIBLE",   0)
    total        = max(n_tx, 1)

    if rate > 5:
        global_risk = "CRITIQUE"
    elif rate > 1:
        global_risk = "ELEVE"
    else:
        global_risk = "FAIBLE"

    # ── Textes descriptifs ────────────────────────────────────────────────────
    resume_text = (
        "Notre systeme d'intelligence artificielle a analyse l'ensemble des transactions "
        "soumises et identifie les operations dont les caracteristiques s'ecartent "
        "significativement des comportements normaux. Les resultats sont presentes en "
        "trois categories : transactions normales, a surveiller et critiques. "
        f"Sur {n_tx:,} transactions analysees, {n_fraud:,} ont ete signalees comme suspectes, "
        f"representant {rate:.2f}% du portefeuille et un montant total de "
        f"{amount_risk:,.2f} TND exposes."
    )

    analyse_text = (
        "Les graphiques ci-dessous permettent de visualiser rapidement la distribution "
        "des transactions, les niveaux de risque detectes et l'exposition financiere "
        "par type d'operation. Chaque graphique est accompagne de son interpretation "
        "pour faciliter la prise de decision de l'auditeur."
    )

    transaction_text = (
        "Le tableau ci-dessous presente les 10 transactions les plus suspectes, "
        "classees par indice de risque decroissant. Les transactions marquees 'ALERTE' "
        "necessitent une verification immediate. L'indice IA varie de 0 (transaction "
        "normale) a 1 (fraude quasi-certaine)."
    )

    # ── Remplacement des placeholders ────────────────────────────────────────
    doc = Document(str(TEMPLATE_PATH))

    # Trouver les paragraphes de reference AVANT modification
    para_tx_text    = _find_para(doc, "{{transaction_text}}")
    para_section04  = _find_para(doc, "04 Fiche")
    para_section05  = _find_para(doc, "05 Recommandations")
    para_section06  = _find_para(doc, "06 Glossaire")

    mapping = {
        "{{date}}":                  date_str,
        "{{resume_text}}":           resume_text,
        "{{analyse_text}}":          analyse_text,
        "{{transaction_text}}":      transaction_text,
        "{{taux_de_fraudes}}":       f"{rate:.2f}%",
        "{{anomalies_détéctées}}": f"{n_fraud:,}",
        "{{transactions_analysées}}": f"{n_tx:,}",
        "{{montant_exposé}}":   _fmt_amount(amount_risk),
        "{{pourcentage}}":           f"{rate:.2f}%",
        "{{remarque}}":              RISK_LABEL[global_risk].upper(),
        "{{pourcentage_faible}}":    f"{n_faible / total * 100:.0f}%",
        "{{pourcentage_elevé}}": f"{n_eleve / total * 100:.0f}%",
        "{{pourcentage_critique}}":  f"{n_critique / total * 100:.0f}%",
    }
    _replace_all(doc, mapping)

    # ── Graphiques ─────────────────────────────────────────────────────────────
    buf_donut = _chart_donut(n_tx, n_fraud)
    buf_bars  = _chart_risk_bars(transactions)
    buf_types = _chart_amount_type(transactions)

    # Cherche par sous-chaine robuste (ignore accents / espaces insecables)
    _insert_chart_after_label(doc, "paration des transactions", buf_donut, width=4.8)
    _insert_chart_after_label(doc, "isque d",                   buf_bars,  width=4.8)
    _insert_chart_after_label(doc, "position financi",          buf_types, width=6.5)

    if explain_results:
        buf_shap = _chart_shap(explain_results)
        if buf_shap:
            _insert_chart_after_label(doc, "Facteurs d",  buf_shap, width=6.5)

    # ── Tableau transactions ──────────────────────────────────────────────────
    if para_tx_text and transactions:
        top10 = sorted(transactions, key=lambda t: t.get("xgb_score", 0), reverse=True)[:10]
        headers = ["#", "Identifiant", "Type", "Montant (TND)", "Indice IA", "Statut"]
        widths  = [0.25, 1.10, 0.85, 1.10, 0.95, 1.15]
        rows = []
        for rank, tx in enumerate(top10, 1):
            risk = tx.get("risk_level", "FAIBLE")
            icon = {"CRITIQUE": "ALERTE", "ELEVE": "ATTENTION", "FAIBLE": "OK"}[risk]
            rows.append([
                str(rank),
                str(tx.get("tx_id", ""))[:14],
                str(tx.get("type", "-")),
                f"{tx.get('amount', 0):,.0f}",
                f"{tx.get('xgb_score', 0):.3f}",
                icon,
            ])
        _add_table_after_para(doc, para_tx_text, headers, rows, col_widths=widths)
        _clear_empty_paras_after(doc, para_tx_text, count=9)

    # ── Fiches critiques ──────────────────────────────────────────────────────
    explains_map: dict[str, dict] = {}
    if explain_results:
        for e in explain_results:
            key = str(e.get("tx_id", e.get("id", "")))
            explains_map[key] = e

    critique_txs = [t for t in transactions if t.get("risk_level") == "CRITIQUE"]

    if para_section04:
        _clear_empty_paras_after(doc, para_section04, count=9)

        if not critique_txs:
            last = _add_para_after(
                para_section04._element,
                "Aucune transaction critique detectee lors de cette analyse.",
                italic=True, size=10, color=RGB_GRAY,
            )
        else:
            last = para_section04._element
            for idx, tx in enumerate(critique_txs[:8], 1):
                tx_id = str(tx.get("tx_id", f"TX-{idx}"))
                expl  = explains_map.get(tx_id, {})
                xgb   = float(tx.get("xgb_score", 0))
                llm   = expl.get("llm", expl)

                # En-tete fiche
                last = _add_para_after(
                    last,
                    f"  Fiche #{idx}  -  Transaction {tx_id}  [ ALERTE ]",
                    bold=True, size=10, color=RGB_WHITE,
                )
                _set_para_bg(last, "C00000")

                # Infos de base
                last = _add_para_after(
                    last,
                    f"Type : {tx.get('type', '-')}     |     Montant : {tx.get('amount', 0):,.2f} TND",
                    size=10,
                )

                # Indice de risque
                bar_filled = "#" * int(xgb * 20)
                bar_empty  = "-" * (20 - int(xgb * 20))
                last = _add_para_after(
                    last,
                    f"Indice de risque IA :  [{bar_filled}{bar_empty}]  {xgb:.3f} / 1.000",
                    size=9, color=RGB_RED,
                )

                # Analyse LLM
                resume = llm.get("resume", "")
                if resume:
                    last = _add_para_after(last, "Analyse :", bold=True, size=10, color=RGB_DARK)
                    last = _add_para_after(last, resume[:450], size=9, color=RGB_GRAY)

                # Raisons
                raisons = llm.get("raisons", [])
                if raisons:
                    last = _add_para_after(
                        last, "Pourquoi cette transaction est suspecte :",
                        bold=True, size=10, color=RGB_DARK,
                    )
                    for r in raisons[:4]:
                        last = _add_para_after(
                            last, f"   -  {str(r)[:130]}", size=9, color=RGB_GRAY,
                        )

                # SHAP features
                shap_vals = expl.get("shap_values", {})
                if not shap_vals and "top_features" in expl:
                    shap_vals = {
                        f["feature"]: f.get("error", f.get("value", 0))
                        for f in expl.get("top_features", [])
                    }
                if shap_vals:
                    top3 = sorted(shap_vals.items(), key=lambda x: abs(float(x[1])), reverse=True)[:3]
                    last = _add_para_after(
                        last, "Facteurs declencheurs (selon l'IA) :",
                        bold=True, size=10, color=RGB_DARK,
                    )
                    for feat, val in top3:
                        feat_fr  = FEATURE_FR.get(feat, feat)
                        sign     = "+" if float(val) >= 0 else ""
                        f_expl   = FEATURE_EXPLAIN.get(feat, "")
                        last = _add_para_after(
                            last,
                            f"   {feat_fr} :  score {sign}{float(val):.4f}",
                            size=9, color=RGB_ORANGE,
                        )
                        if f_expl:
                            last = _add_para_after(
                                last, f"      -> {f_expl}", italic=True, size=9, color=RGB_GRAY,
                            )

                # Actions recommandees
                actions = llm.get("actions_recommandees", expl.get("actions_recommandees", []))
                if actions:
                    last = _add_para_after(
                        last, "Actions recommandees pour l'auditeur :",
                        bold=True, size=10, color=RGB_ORANGE,
                    )
                    for act in actions[:3]:
                        last = _add_para_after(
                            last, f"   [>]  {str(act)[:135]}", size=9, color=RGB_GRAY,
                        )

                # Separateur
                last = _add_para_after(last, "", size=6)

    # ── Recommandations ───────────────────────────────────────────────────────
    if para_section05:
        _clear_empty_paras_after(doc, para_section05, count=9)

        recs = []
        if n_critique > 0:
            recs.append(
                f"Investiguer immediatement les {n_critique} transaction(s) marquee(s) 'CRITIQUE'. "
                "Verifiez la legitimite des beneficiaires et les pieces justificatives associees."
            )
        if n_eleve > 0:
            recs.append(
                f"Planifier une revue approfondie des {n_eleve} transaction(s) 'A SURVEILLER' "
                "dans les 48 heures suivantes."
            )
        if amount_risk > 0:
            recs.append(
                f"Le montant total expose est de {amount_risk:,.2f} TND. "
                "Evaluez l'impact financier potentiel avec le responsable financier."
            )
        recs += [
            "Documenter toutes vos verifications dans le dossier d'audit permanent.",
            "En cas de fraude confirmee, declencher la procedure d'alerte interne.",
            "Conserver ce rapport joint aux preuves collectees dans le dossier d'audit.",
        ]

        last = para_section05._element
        for i, rec in enumerate(recs, 1):
            last = _add_para_after(last, f"  {i}.  {rec}", size=10, color=RGB_GRAY)

    # ── Glossaire ─────────────────────────────────────────────────────────────
    if para_section06:
        _clear_empty_paras_after(doc, para_section06, count=3)

        last = para_section06._element
        for term, definition in GLOSSAIRE:
            last = _add_para_after(last, f"* {term} :", bold=True, size=10, color=RGB_DARK)
            last = _add_para_after(last, f"   {definition}", size=9, color=RGB_GRAY)
            last = _add_para_after(last, "", size=5)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _set_para_bg(p_elem, hex_color: str) -> None:
    """Set paragraph background via shading on paragraph properties."""
    # Trouve ou cree w:pPr
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    pPr = p_elem.find(f"{{{ns}}}pPr")
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        p_elem.insert(0, pPr)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"),  hex_color)
    shd.set(qn("w:color"), hex_color)
    shd.set(qn("w:val"),   "clear")
    pPr.append(shd)
