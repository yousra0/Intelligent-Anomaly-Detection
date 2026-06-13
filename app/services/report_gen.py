"""
app/services/report_gen.py
Rapport PDF PwC  - Audit de détection de fraude financière.
Thème PwC officiel, langage accessible aux auditeurs non-techniques.
"""
from __future__ import annotations

import io
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

# ─── Charte graphique PwC ────────────────────────────────────────────────────
PWC_ORANGE  = (208, 74,  2)
PWC_DARK    = (41,  56,  84)
GRAY        = (80,  80,  80)
LIGHT_BG    = (248, 248, 248)
WHITE       = (255, 255, 255)
BLACK       = (30,  30,  30)
RED_ALERT   = (192, 0,   0)
ORANGE_WARN = (230, 110, 0)
GREEN_OK    = (0,   130, 70)

_C_ORANGE = "#D04A02"
_C_DARK   = "#293854"
_C_RED    = "#C00000"
_C_WARN   = "#E66E00"
_C_GREEN  = "#008246"
_C_LGRAY  = "#F8F8F8"
_C_MGRAY  = "#B0B0B0"

# ─── Traductions et libellés ──────────────────────────────────────────────────
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
    "oldbalanceOrg":     "Solde avant (emetteur)",
    "newbalanceOrig":    "Solde apres (emetteur)",
    "oldbalanceDest":    "Solde avant (destinataire)",
    "newbalanceDest":    "Solde apres (destinataire)",
    "day_of_week":       "Jour de la semaine",
    "week":              "Numero de semaine",
}

FEATURE_EXPLAIN = {
    "balance_diff_orig": (
        "Le solde du compte emetteur a change d'une facon qui ne correspond pas aux "
        "operations declarees. C'est l'indicateur le plus fort de fraude potentielle."
    ),
    "log_amount": (
        "Le montant de cette transaction est inhabituellement eleve par rapport aux "
        "transactions similaires dans la base de donnees."
    ),
    "hour": (
        "La transaction a eu lieu a une heure atypique (nuit ou tres tot le matin), "
        "periode connue pour concentrer les activites frauduleuses."
    ),
    "dest_zero_balance": (
        "Le compte destinataire affiche un solde nul apres reception des fonds. "
        "Cela est caracteristique des comptes 'mule' utilises pour blanchir de l'argent."
    ),
    "type_CASH_OUT": (
        "Les retraits especes sont la methode la plus frequente dans les fraudes detectees."
    ),
    "type_TRANSFER": (
        "Les virements de grande valeur vers des comptes inconnus sont a haut risque."
    ),
}

RISK_LABEL = {
    "CRITIQUE": "A investiguer immediatement",
    "ELEVE":    "A surveiller prioritairement",
    "FAIBLE":   "Aucun probleme detecte",
}

RISK_COLOR = {
    "CRITIQUE": RED_ALERT,
    "ELEVE":    ORANGE_WARN,
    "FAIBLE":   GREEN_OK,
}

RISK_ICON = {
    "CRITIQUE": "ALERTE",
    "ELEVE":    "ATTENTION",
    "FAIBLE":   "OK",
}


# ─── Helpers matplotlib ───────────────────────────────────────────────────────

def _to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def _chart_donut(n_tx: int, n_fraud: int) -> io.BytesIO:
    n_ok = max(n_tx - n_fraud, 0)
    sizes  = [n_ok, n_fraud] if n_fraud > 0 else [1, 0]
    labels = [f"Normales\n{n_ok:,}", f"Suspectes\n{n_fraud:,}"]
    colors = [_C_DARK, _C_ORANGE]

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    fig.patch.set_facecolor("white")
    wedges, _, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=7.5),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(7.5)
    ax.set_title("Repartition des transactions", fontsize=8.5,
                 fontweight="bold", color=_C_DARK, pad=8)
    plt.tight_layout(pad=0.3)
    return _to_buf(fig)


def _chart_risk_bars(transactions: list[dict]) -> io.BytesIO:
    counts = Counter(t.get("risk_level", "FAIBLE") for t in transactions)
    cats   = ["FAIBLE", "ELEVE", "CRITIQUE"]
    labels = ["Normale", "A surveiller", "Critique"]
    vals   = [counts.get(c, 0) for c in cats]
    hues   = [_C_GREEN, _C_WARN, _C_RED]
    max_v  = max(vals) if max(vals) > 0 else 1

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_C_LGRAY)
    bars = ax.bar(labels, vals, color=hues, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.03,
                str(v), ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=_C_DARK,
            )
    ax.set_title("Niveau de risque detecte", fontsize=8.5,
                 fontweight="bold", color=_C_DARK, pad=8)
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

    items = sorted(sums.items(), key=lambda x: -x[1])[:6]
    if not items:
        items = [("Aucune donnee", 0)]

    types = [i[0] for i in items]
    amts  = [i[1] / 1_000 for i in items]
    pal   = [_C_ORANGE, _C_DARK, _C_RED, _C_WARN, _C_GREEN, _C_MGRAY]
    max_a = max(amts) if max(amts) > 0 else 1

    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_C_LGRAY)
    bars = ax.barh(types[::-1], amts[::-1],
                   color=pal[:len(types)][::-1], edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, amts[::-1]):
        ax.text(
            bar.get_width() + max_a * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,.0f}k TND", va="center", ha="left", fontsize=7.5, color=_C_DARK,
        )
    ax.set_title("Exposition financiere par type d'operation (kTND)",
                 fontsize=8.5, fontweight="bold", color=_C_DARK, pad=8)
    ax.set_xlabel("Milliers TND", fontsize=7, color="#666666")
    ax.set_xlim(0, max_a * 1.35)
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.3)
    return _to_buf(fig)


def _chart_shap_global(explain_results: list[dict]) -> io.BytesIO | None:
    totals: dict[str, float] = defaultdict(float)
    counts_s: dict[str, int] = defaultdict(int)
    for e in explain_results:
        for f, v in e.get("shap_values", {}).items():
            totals[f] += abs(float(v))
            counts_s[f] += 1
    if not totals:
        return None

    mean_abs = {f: totals[f] / counts_s[f] for f in totals}
    items  = sorted(mean_abs.items(), key=lambda x: -x[1])[:8]
    labels = [FEATURE_FR.get(f, f) for f, _ in items]
    vals   = [v for _, v in items]
    thresh = vals[0] * 0.35 if vals else 0
    hues   = [_C_ORANGE if v >= thresh else _C_DARK for v in vals]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_C_LGRAY)
    ax.barh(labels[::-1], vals[::-1], color=hues[::-1], edgecolor="white", linewidth=1)
    ax.set_title("Facteurs les plus declencheurs d'alertes (importance SHAP)",
                 fontsize=8.5, fontweight="bold", color=_C_DARK, pad=8)
    ax.set_xlabel("Importance moyenne", fontsize=7, color="#666666")
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.3)
    return _to_buf(fig)


def _fmt_amount(amount: float) -> str:
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"{amount / 1_000:.0f}k"
    return f"{amount:,.0f}"


# ─── Classe PDF ───────────────────────────────────────────────────────────────

class PwCReport(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(left=15, top=18, right=15)

    def header(self):
        self.set_fill_color(*PWC_ORANGE)
        self.rect(0, 0, 210, 11, "F")
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*WHITE)
        self.set_xy(0, 2)
        self.cell(0, 7, "PwC Tunisie  |  Departement Audit & Assurance", align="C")
        self.set_text_color(*BLACK)
        self.ln(5)

    def footer(self):
        self.set_y(-11)
        self.set_fill_color(*PWC_DARK)
        self.rect(0, 286, 210, 11, "F")
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*WHITE)
        self.set_xy(15, 287.5)
        self.cell(90, 6, "CONFIDENTIEL  - Usage exclusif audit interne", align="L")
        self.set_xy(105, 287.5)
        self.cell(90, 6, f"Page {self.page_no()}", align="R")
        self.set_text_color(*BLACK)

    def section_title(self, title: str, color: tuple = None):
        c = color or PWC_DARK
        self.set_fill_color(*c)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8.5, f"   {title}", ln=True, fill=True)
        self.set_text_color(*BLACK)
        self.ln(2)

    def body(self, text: str, size: int = 9):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*GRAY)
        self.multi_cell(0, 5.2, text)
        self.set_text_color(*BLACK)

    def lv(self, label: str, value: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*PWC_DARK)
        self.cell(72, 5.5, label)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.cell(0, 5.5, value, ln=True)
        self.set_text_color(*BLACK)

    def draw_progress_bar(self, x: float, y: float, w: float, h: float,
                          ratio: float, color: tuple = None):
        color = color or PWC_ORANGE
        filled = w * min(max(ratio, 0), 1)
        self.set_fill_color(220, 220, 220)
        self.rect(x, y, w, h, "F")
        self.set_fill_color(*color)
        if filled > 0:
            self.rect(x, y, filled, h, "F")
        self.set_draw_color(180, 180, 180)
        self.rect(x, y, w, h, "D")
        self.set_draw_color(0, 0, 0)


# ─── Générateur principal ─────────────────────────────────────────────────────

def generate_pwc_report(
    predict_result: dict,
    explain_results: list[dict] | None = None,
) -> bytes:
    now       = datetime.now()
    date_str  = now.strftime("%d/%m/%Y a %H:%M")

    n_tx         = predict_result.get("n_transactions", 0)
    n_fraud      = predict_result.get("n_fraud", 0)
    rate         = predict_result.get("fraud_rate_pct", 0.0)
    amount_risk  = predict_result.get("amount_at_risk", 0.0)
    model_used   = predict_result.get("model_used", "XGB_smote")
    threshold    = predict_result.get("threshold", 0.355)
    transactions = predict_result.get("transactions", [])

    if rate > 5:
        global_risk, risk_col = "CRITIQUE", RED_ALERT
    elif rate > 1:
        global_risk, risk_col = "ELEVE", ORANGE_WARN
    else:
        global_risk, risk_col = "FAIBLE", GREEN_OK

    risk_counts = Counter(t.get("risk_level", "FAIBLE") for t in transactions)

    pdf = PwCReport()
    pdf.set_title("Rapport Audit Fraude - PwC Tunisie")
    pdf.set_author("PwC Tunisie")

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1  - Couverture
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Grande bande orange
    pdf.set_fill_color(*PWC_ORANGE)
    pdf.rect(0, 14, 210, 56, "F")

    pdf.set_xy(15, 20)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*WHITE)
    pdf.cell(180, 13, "RAPPORT D'AUDIT", align="C", ln=True)

    pdf.set_xy(15, 34)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(180, 9, "Detection de Fraude Financiere", align="C", ln=True)

    pdf.set_xy(15, 46)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(180, 7, f"Genere le {date_str}", align="C", ln=True)

    pdf.set_text_color(*BLACK)

    # Bandeau niveau de risque global
    r, g, b = risk_col
    pdf.set_fill_color(r, g, b)
    pdf.rect(15, 76, 180, 10, "F")
    pdf.set_xy(15, 77)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*WHITE)
    label_global = f"Niveau de risque global : {RISK_LABEL[global_risk].upper()}"
    pdf.cell(180, 8, label_global, align="C")
    pdf.set_text_color(*BLACK)

    # Intro pour l'auditeur
    pdf.set_y(94)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*PWC_DARK)
    pdf.cell(0, 6, "A l'attention de l'auditeur financier", align="C", ln=True)
    pdf.set_text_color(*BLACK)
    pdf.ln(2)
    pdf.body(
        "Ce document presente les resultats de l'analyse automatisee des transactions "
        "financieres par notre systeme d'intelligence artificielle. Il a ete concu pour "
        "etre lu et compris par tout auditeur, independamment de ses connaissances "
        "techniques. Chaque anomalie detectee est expliquee en termes clairs, avec les "
        "actions recommandees. En cas de doute, contactez l'equipe Audit IT PwC.",
        size=9,
    )

    # Ligne déco
    pdf.ln(8)
    pdf.set_fill_color(*PWC_ORANGE)
    pdf.rect(15, pdf.get_y(), 180, 1.2, "F")
    pdf.ln(7)

    # Infos d'analyse
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*PWC_DARK)
    pdf.cell(0, 5.5, "Informations sur l'analyse", ln=True)
    pdf.ln(1)
    pdf.lv("Transactions analysees :", f"{n_tx:,}")
    pdf.lv("Date d'analyse :", date_str)
    pdf.lv("Modele IA utilise :", "XGBoost + AutoEncoder (intelligence artificielle)")
    pdf.lv("Seuil de detection :", f"{threshold:.3f}  (calibre sur donnees historiques)")

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "CONFIDENTIEL  - Ce document est destine a l'usage exclusif de l'equipe d'audit.", align="C", ln=True)
    pdf.set_text_color(*BLACK)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2  - Résumé exécutif + KPI + Jauge
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("1. RESUME EXECUTIF")

    pdf.body(
        "Notre systeme d'intelligence artificielle a analyse l'ensemble des transactions "
        "soumises et identifie les operations dont les caracteristiques s'ecartent "
        "significativement des comportements normaux. Les resultats sont presentes en "
        "trois categories : transactions normales, a surveiller et critiques.",
        size=9,
    )
    pdf.ln(4)

    # ── 4 boîtes KPI ──────────────────────────────────────────────────────────
    y0    = pdf.get_y()
    bw    = 42
    bh    = 28
    gap   = 4
    xs    = [15 + i * (bw + gap) for i in range(4)]

    kpi_data = [
        ("Transactions analysees",  f"{n_tx:,}",                  "",                  PWC_DARK),
        ("Anomalies detectees",     f"{n_fraud:,}",               f"sur {n_tx:,}",      RED_ALERT  if n_fraud > 0 else GREEN_OK),
        ("Taux de fraude",          f"{rate:.2f}%",               "du portefeuille",    ORANGE_WARN if rate > 1   else GREEN_OK),
        ("Montant expose",          _fmt_amount(amount_risk),     "TND",                RED_ALERT  if amount_risk > 100_000 else ORANGE_WARN),
    ]

    for x, (lbl, val, sub, col) in zip(xs, kpi_data):
        rr, gg, bb = col
        pdf.set_fill_color(rr, gg, bb)
        pdf.rect(x, y0, bw, bh, "F")
        pdf.set_fill_color(*PWC_ORANGE)
        pdf.rect(x, y0 + bh - 2, bw, 2, "F")

        pdf.set_xy(x, y0 + 2)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*WHITE)
        pdf.cell(bw, 4.5, lbl, align="C")

        pdf.set_xy(x, y0 + 8)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(bw, 8, val, align="C")

        if sub:
            pdf.set_xy(x, y0 + 17)
            pdf.set_font("Helvetica", "", 6)
            pdf.cell(bw, 4, sub, align="C")

        pdf.set_text_color(*BLACK)

    pdf.set_y(y0 + bh + 8)

    # ── Jauge de risque global ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*PWC_DARK)
    pdf.cell(0, 5, "Indicateur de risque global :", ln=True)
    pdf.ln(1)

    gy       = pdf.get_y()
    seg_data = [
        ("FAIBLE  (< 1 %)",  0.40, GREEN_OK),
        ("ELEVE   (1-5 %)",  0.35, ORANGE_WARN),
        ("CRITIQUE (> 5 %)", 0.25, RED_ALERT),
    ]
    gx = 15
    for seg_lbl, seg_ratio, seg_col in seg_data:
        seg_w = 180 * seg_ratio
        pdf.set_fill_color(*seg_col)
        pdf.rect(gx, gy, seg_w, 8, "F")
        pdf.set_xy(gx, gy)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*WHITE)
        pdf.cell(seg_w, 8, seg_lbl, align="C")
        pdf.set_text_color(*BLACK)
        gx += seg_w

    # Marqueur triangulaire (petit rectangle coloré)
    rate_capped = min(rate, 10.0)
    if rate_capped <= 1.0:
        marker_frac = 0.40 * (rate_capped / 1.0) * 0.9
    elif rate_capped <= 5.0:
        marker_frac = 0.40 + 0.35 * ((rate_capped - 1.0) / 4.0)
    else:
        marker_frac = 0.75 + 0.25 * min((rate_capped - 5.0) / 5.0, 1.0)

    mx = 15 + 180 * marker_frac
    mr, mg, mb = risk_col
    pdf.set_fill_color(mr, mg, mb)
    pdf.rect(mx - 2, gy + 8, 4, 3, "F")

    pdf.set_y(gy + 14)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(mr, mg, mb)
    pdf.cell(0, 5, f"Position actuelle : {rate:.2f}%  - {RISK_LABEL[global_risk].upper()}", ln=True)
    pdf.set_text_color(*BLACK)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3  - Graphiques
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("2. ANALYSE VISUELLE DES RESULTATS")

    pdf.body(
        "Les graphiques ci-dessous permettent de visualiser rapidement la distribution "
        "des transactions, les niveaux de risque detectes et l'exposition financiere "
        "par type d'operation.",
        size=9,
    )
    pdf.ln(3)

    buf_donut = _chart_donut(n_tx, n_fraud)
    buf_bars  = _chart_risk_bars(transactions)
    buf_types = _chart_amount_type(transactions)

    # Ligne 1 : donut + barres côte à côte
    y_row1 = pdf.get_y()
    pdf.image(buf_donut, x=15,   y=y_row1, w=84)
    pdf.image(buf_bars,  x=111,  y=y_row1, w=84)
    pdf.set_y(y_row1 + 62)

    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*GRAY)
    pdf.set_x(15)
    pdf.cell(84, 4, "Fig. 1 - Proportion transactions suspectes / normales", align="C")
    pdf.set_x(111)
    pdf.cell(84, 4, "Fig. 2 - Repartition par niveau de risque", align="C")
    pdf.set_text_color(*BLACK)
    pdf.ln(6)

    # Ligne 2 : exposition financière (pleine largeur)
    y_row2 = pdf.get_y()
    pdf.image(buf_types, x=15, y=y_row2, w=180)
    pdf.set_y(y_row2 + 60)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 4, "Fig. 3 - Exposition financiere (transactions suspectes) par type d'operation", align="C")
    pdf.set_text_color(*BLACK)
    pdf.ln(5)

    # SHAP si disponible
    if explain_results:
        buf_shap = _chart_shap_global(explain_results)
        if buf_shap:
            if pdf.get_y() > 215:
                pdf.add_page()
            y_shap = pdf.get_y()
            pdf.image(buf_shap, x=15, y=y_shap, w=180)
            pdf.set_y(y_shap + 55)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*GRAY)
            pdf.cell(0, 4, "Fig. 4 - Facteurs les plus declencheurs d'alertes (analyse SHAP)", align="C")
            pdf.set_text_color(*BLACK)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 4  - Top 10 transactions
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("3. TRANSACTIONS PRIORITAIRES A CONTROLER")

    pdf.body(
        "Le tableau ci-dessous presente les 10 transactions les plus suspectes, "
        "classees par indice de risque decroissant. Les transactions marquees 'ALERTE' "
        "necessitent une verification immediate.",
        size=9,
    )
    pdf.ln(3)

    cols   = ["#", "Identifiant", "Type", "Montant (TND)", "Indice IA", "Statut"]
    widths = [8, 35, 28, 35, 28, 40]

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*PWC_DARK)
    pdf.set_text_color(*WHITE)
    for h, w in zip(cols, widths):
        pdf.cell(w, 7.5, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(*BLACK)

    top10 = sorted(transactions, key=lambda t: t.get("xgb_score", 0), reverse=True)[:10]

    for rank, tx in enumerate(top10, 1):
        risk = tx.get("risk_level", "FAIBLE")
        rr, gg, bb = RISK_COLOR[risk]
        alt = (rank % 2 == 0)
        bg  = LIGHT_BG if alt else WHITE
        pdf.set_fill_color(*bg)

        row_data = [
            str(rank),
            str(tx.get("tx_id", ""))[:14],
            str(tx.get("type", " -")),
            f"{tx.get('amount', 0):,.0f}",
            f"{tx.get('xgb_score', 0):.3f} / 1.000",
        ]
        pdf.set_font("Helvetica", "", 8)
        for val, w in zip(row_data, widths[:-1]):
            pdf.cell(w, 6.5, val, border=1, fill=True, align="C")

        pdf.set_text_color(rr, gg, bb)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.cell(widths[-1], 6.5, RISK_ICON[risk], border=1, fill=True, align="C")
        pdf.set_text_color(*BLACK)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*PWC_DARK)
    pdf.cell(0, 5, "Legende des statuts :", ln=True)
    pdf.ln(1)
    for risk_key, rlabel in RISK_LABEL.items():
        rr, gg, bb = RISK_COLOR[risk_key]
        pdf.set_fill_color(rr, gg, bb)
        pdf.rect(15, pdf.get_y() + 1, 5, 4, "F")
        pdf.set_x(22)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(rr, gg, bb)
        pdf.cell(18, 6, RISK_ICON[risk_key])
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*GRAY)
        pdf.cell(0, 6, f": {rlabel}", ln=True)
    pdf.set_text_color(*BLACK)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGES 5+  - Fiches détaillées critiques
    # ═══════════════════════════════════════════════════════════════════════════
    critique_txs = [t for t in transactions if t.get("risk_level") == "CRITIQUE"]

    explains_map: dict[str, dict] = {}
    if explain_results:
        for e in explain_results:
            key = str(e.get("tx_id", e.get("id", "")))
            explains_map[key] = e

    if critique_txs:
        pdf.add_page()
        pdf.section_title("4. FICHES DE CAS  - ALERTES CRITIQUES", color=RED_ALERT)
        pdf.body(
            "Pour chaque transaction classee 'CRITIQUE', vous trouverez ci-dessous "
            "une fiche explicative detaillant les raisons de l'alerte et les actions "
            "recommandees. Ces fiches sont redigees en langage accessible.",
            size=9,
        )
        pdf.ln(3)

        for idx, tx in enumerate(critique_txs[:8], 1):
            tx_id  = str(tx.get("tx_id", f"TX-{idx}"))
            expl   = explains_map.get(tx_id, {})
            xgb    = float(tx.get("xgb_score", 0))
            ae_s   = float(tx.get("ae_score", 0))

            if pdf.get_y() > 215:
                pdf.add_page()
                pdf.section_title("4. FICHES DE CAS  - ALERTES CRITIQUES (suite)", color=RED_ALERT)
                pdf.ln(2)

            # En-tête fiche
            fy = pdf.get_y()
            pdf.set_fill_color(*RED_ALERT)
            pdf.rect(15, fy, 180, 8, "F")
            pdf.set_xy(15, fy)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*WHITE)
            pdf.cell(148, 8, f"  Fiche #{idx}  - Transaction {tx_id}")
            pdf.cell(32, 8, "[ ALERTE ]", align="C")
            pdf.set_text_color(*BLACK)
            pdf.ln(9)

            # Infos de base
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(*PWC_DARK)
            pdf.cell(55, 5.5, "Type d'operation :")
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(*GRAY)
            pdf.cell(45, 5.5, tx.get("type", " -"))
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(*PWC_DARK)
            pdf.cell(35, 5.5, "Montant :")
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(*RED_ALERT)
            pdf.cell(0, 5.5, f"{tx.get('amount', 0):,.2f} TND", ln=True)
            pdf.set_text_color(*BLACK)

            # Barre de risque IA
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(*PWC_DARK)
            pdf.cell(55, 5.5, "Indice de risque IA :")
            pdf.set_text_color(*BLACK)
            bar_x = pdf.get_x()
            bar_y = pdf.get_y() + 1
            pdf.draw_progress_bar(bar_x, bar_y, 80, 4, xgb, RED_ALERT)
            pdf.set_xy(bar_x + 83, bar_y - 1)
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(*RED_ALERT)
            pdf.cell(0, 5.5, f"{xgb:.3f} / 1.000", ln=True)
            pdf.set_text_color(*BLACK)

            # Analyse LLM
            llm    = expl.get("llm", expl)
            resume = llm.get("resume", "")
            raisons = llm.get("raisons", [])
            actions = llm.get("actions_recommandees", expl.get("actions_recommandees", []))

            if resume:
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(*PWC_DARK)
                pdf.cell(0, 5.5, "Analyse :", ln=True)
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(*GRAY)
                pdf.multi_cell(0, 5, resume[:450])
                pdf.set_text_color(*BLACK)

            if raisons:
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(*PWC_DARK)
                pdf.cell(0, 5.5, "Pourquoi cette transaction est suspecte :", ln=True)
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(*GRAY)
                for r_item in raisons[:4]:
                    pdf.cell(6, 5, "")
                    pdf.cell(0, 5, f"- {str(r_item)[:130]}", ln=True)
                pdf.set_text_color(*BLACK)

            # SHAP features
            shap_vals: dict = expl.get("shap_values", {})
            if not shap_vals and "top_features" in expl:
                shap_vals = {
                    f["feature"]: f.get("error", f.get("value", 0))
                    for f in expl.get("top_features", [])
                }
            if shap_vals:
                top3 = sorted(shap_vals.items(), key=lambda x: abs(float(x[1])), reverse=True)[:3]
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(*PWC_DARK)
                pdf.cell(0, 5.5, "Facteurs declencheurs (selon l'IA) :", ln=True)
                for feat, val in top3:
                    feat_fr  = FEATURE_FR.get(feat, feat)
                    sign     = "+" if float(val) >= 0 else ""
                    f_expl   = FEATURE_EXPLAIN.get(feat, "")
                    pdf.set_font("Helvetica", "B", 8.5)
                    pdf.set_text_color(*PWC_ORANGE)
                    pdf.cell(6, 5, "")
                    pdf.cell(68, 5, f"{feat_fr} :")
                    pdf.set_font("Helvetica", "", 8.5)
                    pdf.set_text_color(*GRAY)
                    pdf.cell(0, 5, f"score {sign}{float(val):.4f}", ln=True)
                    if f_expl:
                        pdf.set_font("Helvetica", "I", 7.5)
                        pdf.set_text_color(140, 140, 140)
                        pdf.cell(6, 4.5, "")
                        pdf.multi_cell(0, 4.5, f"-> {f_expl}")
                pdf.set_text_color(*BLACK)

            if actions:
                pdf.ln(1)
                ay = pdf.get_y()
                pdf.set_fill_color(255, 246, 238)
                pdf.rect(15, ay, 180, 5.5, "F")
                pdf.set_xy(15, ay)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(*PWC_ORANGE)
                pdf.cell(0, 5.5, "  Actions recommandees pour l'auditeur :", ln=True)
                pdf.set_text_color(*BLACK)
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(*GRAY)
                for act in actions[:3]:
                    pdf.cell(6, 5.5, "")
                    pdf.cell(0, 5.5, f"[>]  {str(act)[:135]}", ln=True)
                pdf.set_text_color(*BLACK)

            pdf.ln(4)
            pdf.set_fill_color(*PWC_ORANGE)
            pdf.rect(15, pdf.get_y() - 1.5, 180, 0.8, "F")
            pdf.ln(3)

    # ═══════════════════════════════════════════════════════════════════════════
    # DERNIÈRE PAGE  - Recommandations + Glossaire + Disclaimer
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("5. RECOMMANDATIONS POUR L'AUDITEUR", color=PWC_ORANGE)

    n_critique = risk_counts.get("CRITIQUE", 0)
    n_eleve    = risk_counts.get("ELEVE", 0)

    recs = []
    if n_critique > 0:
        recs.append(
            f"Investiguer immediatement les {n_critique} transaction(s) marquee(s) "
            "'CRITIQUE'. Verifiez la legitimite des beneficiaires et les pieces "
            "justificatives associees."
        )
    if n_eleve > 0:
        recs.append(
            f"Planifier une revue approfondie des {n_eleve} transaction(s) 'A SURVEILLER' "
            "dans les 48 heures suivantes."
        )
    if amount_risk > 0:
        recs.append(
            f"Le montant total expose est de {amount_risk:,.2f} TND. Evaluez l'impact "
            "financier potentiel avec le responsable financier."
        )
    recs += [
        "Documenter toutes vos verifications dans le dossier d'audit permanent.",
        "En cas de fraude confirmee, declencher la procedure d'alerte conformement "
        "a la politique interne de lutte contre la fraude.",
        "Conserver ce rapport joint aux preuves collectees dans le dossier d'audit.",
    ]

    for i, rec in enumerate(recs, 1):
        ry = pdf.get_y()
        pdf.set_fill_color(*PWC_DARK)
        pdf.rect(15, ry + 1, 6, 6, "F")
        pdf.set_xy(15, ry)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*WHITE)
        pdf.cell(6, 8, f" {i}", align="C")
        pdf.set_xy(23, ry)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*GRAY)
        pdf.multi_cell(0, 5.5, rec)
        if pdf.get_y() < ry + 8:
            pdf.set_y(ry + 8)
        pdf.ln(1)
    pdf.set_text_color(*BLACK)

    # Glossaire
    pdf.ln(4)
    pdf.section_title("6. GLOSSAIRE  - TERMES TECHNIQUES EXPLIQUES")

    glossaire = [
        ("Intelligence artificielle (IA)",
         "Programme informatique capable d'apprendre a partir de donnees historiques "
         "pour detecter des comportements anormaux, sans etre explicitement programme."),
        ("XGBoost",
         "Algorithme d'apprentissage automatique qui analyse de nombreux indicateurs "
         "simultanement pour attribuer un score de risque (0 = normal, 1 = fraude certaine)."),
        ("AutoEncoder",
         "Reseau de neurones artificiel qui apprend a 'reconstruire' des transactions normales. "
         "Quand il echoue, cela signale une anomalie."),
        ("SHAP  - Facteurs declencheurs",
         "Technique qui explique pourquoi l'IA a signale une transaction, en identifiant "
         "les indicateurs qui ont le plus contribue a l'alerte."),
        ("Indice de risque (0 a 1)",
         "Score attribue par l'IA. Plus proche de 1 = plus suspect. Le seuil de "
         f"declenchement est fixe a {threshold:.3f}."),
        ("Solde nul destinataire",
         "Compte recevant l'argent avec solde nul avant la transaction. "
         "Caracteristique souvent associee aux comptes utilises pour blanchiment."),
        ("Ecart de solde",
         "Difference entre le montant debite du compte source et la variation de solde "
         "observee. Un ecart important peut indiquer une manipulation des donnees."),
    ]

    for term, definition in glossaire:
        if pdf.get_y() > 255:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*PWC_DARK)
        pdf.cell(4, 5.5, "")
        pdf.cell(0, 5.5, f"* {term} :", ln=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*GRAY)
        pdf.cell(8, 5, "")
        pdf.multi_cell(0, 5, definition)
        pdf.ln(1)
    pdf.set_text_color(*BLACK)

    # Avertissement légal
    pdf.ln(4)
    if pdf.get_y() > 248:
        pdf.add_page()
    legal_y = pdf.get_y()
    pdf.set_fill_color(*LIGHT_BG)
    pdf.rect(15, legal_y, 180, 22, "F")
    pdf.set_fill_color(*PWC_ORANGE)
    pdf.rect(15, legal_y, 3, 22, "F")
    pdf.set_xy(20, legal_y + 2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*PWC_DARK)
    pdf.cell(0, 5, "Avertissement important", ln=True)
    pdf.set_xy(20, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(
        172, 5,
        "Ce rapport est genere automatiquement par le systeme de detection de fraude "
        "(XGBoost + AutoEncoder). Les resultats constituent des indications et doivent "
        "etre valides par un auditeur qualifie avant toute action. PwC Tunisie ne saurait "
        "etre tenu responsable des decisions prises sur la seule base de ce document.",
    )
    pdf.set_text_color(*BLACK)

    buf = io.BytesIO()
    buf.write(pdf.output())
    return buf.getvalue()
