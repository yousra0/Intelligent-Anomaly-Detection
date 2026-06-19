"""
app/services/column_mapper.py

Moteur de détection sémantique des colonnes CSV.

Principe :
  Pour chaque champ canonique transactionnel (step, type, amount, ...), le moteur
  calcule un score de confiance pour chaque colonne du CSV entrant, en
  combinant quatre niveaux de matching :
    1. Alias exact    (confidence 1.00) – liste exhaustive de noms connus
    2. Alias normalisé (confidence 0.95) – insensible à la casse, tirets, underscores
    3. Signaux composés (confidence 0.80) – présence conjointe de tokens sémantiques
    4. Fuzzy matching  (confidence 0.60) – SequenceMatcher sur les alias normalisés

  L'affectation finale est résolue par une procédure greedy : la paire
  (champ, colonne) de score le plus élevé est retenue en premier, puis
  exclue des candidats restants — aucune colonne n'est affectée deux fois.

Résultat :
  MappingResult.df          → DataFrame avec colonnes renommées en noms canoniques
  MappingResult.mapping     → dict {canonical: original_col}
  MappingResult.confidence  → dict {canonical: score 0..1}
  MappingResult.warnings    → liste de messages pour les mappings incertains
  MappingResult.unmapped    → champs requis non trouvés
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Définition des champs sémantiques
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticField:
    canonical: str
    description: str
    required: bool
    # Liste de noms exacts connus (originaux, pas normalisés)
    exact_aliases: list[str]
    # Tokens qui, ensemble, désignent ce champ de manière non ambiguë
    # Chaque tuple = (tokens "temporels-ou-montant", tokens "sens")
    # Une colonne valide doit contenir AU MOINS UN token de chaque groupe
    compound_signals: list[list[list[str]]]  # OU extérieur, ET intérieur
    # Tokens qui DISQUALIFIENT ce champ si présents
    forbidden_tokens: list[str] = field(default_factory=list)


# Valeurs acceptées pour la colonne `type`
TYPE_VALUE_MAP: dict[str, str] = {
    "transfer"      : "TRANSFER",
    "virement"      : "TRANSFER",
    "cash_out"      : "CASH_OUT",
    "cashout"       : "CASH_OUT",
    "retrait"       : "CASH_OUT",
    "withdrawal"    : "CASH_OUT",
    "cash_in"       : "CASH_IN",
    "cashin"        : "CASH_IN",
    "depot"         : "CASH_IN",
    "dépôt"         : "CASH_IN",
    "deposit"       : "CASH_IN",
    "debit"         : "DEBIT",
    "payment"       : "PAYMENT",
    "paiement"      : "PAYMENT",
    "payement"      : "PAYMENT",
    "pay"           : "PAYMENT",
}


SEMANTIC_FIELDS: dict[str, SemanticField] = {

    "step": SemanticField(
        canonical="step",
        description="Étape temporelle / horodatage",
        required=True,
        exact_aliases=[
            "step", "steps", "time_step", "timestep", "period", "periode",
            "timestamp", "datetime", "date", "date_heure", "heure", "time",
            "t", "idx", "index_time",
        ],
        compound_signals=[
            [["step", "time", "period", "ts"], []],   # n'importe lequel suffit
        ],
        forbidden_tokens=["bal", "solde", "amount", "montant"],
    ),

    "type": SemanticField(
        canonical="type",
        description="Type de transaction",
        required=True,
        exact_aliases=[
            "type", "transaction_type", "type_transaction", "tx_type",
            "txn_type", "trans_type", "category", "categorie", "kind",
            "nature", "operation", "op_type", "payment_type",
            "transaction_category",
        ],
        compound_signals=[
            [["type", "kind", "categ", "natur", "operat"], []],
        ],
        forbidden_tokens=["bal", "solde", "amount", "montant", "name", "nom"],
    ),

    "amount": SemanticField(
        canonical="amount",
        description="Montant de la transaction",
        required=True,
        exact_aliases=[
            "amount", "montant", "amt", "value", "valeur",
            "sum", "somme", "total", "transaction_amount",
            "montant_transaction", "transaction_value", "tx_amount",
            "txn_amount", "trans_amount", "price", "prix",
            "transaction_sum", "montant_tx",
        ],
        compound_signals=[
            [["amount", "montant", "amt", "value", "valeur", "sum", "somme", "price", "prix"], []],
        ],
        forbidden_tokens=["bal", "solde", "old", "new", "avant", "apres", "after", "before"],
    ),

    "nameOrig": SemanticField(
        canonical="nameOrig",
        description="Identifiant compte source (émetteur)",
        required=False,
        exact_aliases=[
            "nameorig", "name_orig", "nameoriginal", "from", "source",
            "sender", "emetteur", "origine", "expediteur", "account_from",
            "account_source", "src_account", "compte_source", "compte_emetteur",
            "payer", "debitor", "compte_debiteur",
        ],
        compound_signals=[
            [["name", "account", "compte", "id", "ref"], ["orig", "source", "src", "from", "emet", "send"]],
        ],
        forbidden_tokens=["bal", "solde", "balance", "amount", "montant"],
    ),

    "oldbalanceOrg": SemanticField(
        canonical="oldbalanceOrg",
        description="Solde avant transaction (compte source)",
        required=True,
        exact_aliases=[
            "oldbalanceorg", "oldbalanceorig", "old_balance_org",
            "old_balance_orig", "balance_before", "balance_before_source",
            "solde_avant", "solde_avant_source", "solde_initial_source",
            "balance_orig_before", "initial_balance", "initial_balance_org",
            "prev_balance", "previous_balance", "balance_origin",
            "old_balance_source", "former_balance",
        ],
        compound_signals=[
            [
                ["old", "before", "avant", "initial", "prev", "previous", "former", "init"],
                ["bal", "solde", "balance"],
                ["orig", "org", "source", "src", "from", "emet", "send"],
            ],
        ],
        forbidden_tokens=["dest", "new", "after", "apres"],
    ),

    "newbalanceOrig": SemanticField(
        canonical="newbalanceOrig",
        description="Solde après transaction (compte source)",
        required=True,
        exact_aliases=[
            "newbalanceorig", "newbalanceorg", "new_balance_orig",
            "new_balance_org", "balance_after", "balance_after_source",
            "solde_apres", "solde_apres_source", "solde_final_source",
            "balance_orig_after", "final_balance", "final_balance_org",
            "after_balance", "post_balance", "new_balance_source",
        ],
        compound_signals=[
            [
                ["new", "after", "apres", "final", "post", "current", "updated"],
                ["bal", "solde", "balance"],
                ["orig", "org", "source", "src", "from", "emet", "send"],
            ],
        ],
        forbidden_tokens=["dest", "old", "before", "avant"],
    ),

    "nameDest": SemanticField(
        canonical="nameDest",
        description="Identifiant compte destination (destinataire)",
        required=False,
        exact_aliases=[
            "namedest", "name_dest", "to", "dest", "destination",
            "recipient", "destinataire", "beneficiaire", "beneficiary",
            "account_to", "account_dest", "dest_account", "compte_dest",
            "compte_destinataire", "payee", "creditor",
        ],
        compound_signals=[
            [["name", "account", "compte", "id", "ref"], ["dest", "to", "recip", "benef", "credit"]],
        ],
        forbidden_tokens=["bal", "solde", "balance", "amount", "montant"],
    ),

    "oldbalanceDest": SemanticField(
        canonical="oldbalanceDest",
        description="Solde avant transaction (compte destination)",
        required=True,
        exact_aliases=[
            "oldbalancedest", "old_balance_dest", "balance_dest_before",
            "solde_dest_avant", "initial_balance_dest", "prev_balance_dest",
            "balance_before_dest", "former_balance_dest",
        ],
        compound_signals=[
            [
                ["old", "before", "avant", "initial", "prev", "previous", "former", "init"],
                ["bal", "solde", "balance"],
                ["dest", "to", "recip", "benef"],
            ],
        ],
        forbidden_tokens=["orig", "org", "source", "src", "new", "after", "apres"],
    ),

    "newbalanceDest": SemanticField(
        canonical="newbalanceDest",
        description="Solde après transaction (compte destination)",
        required=True,
        exact_aliases=[
            "newbalancedest", "new_balance_dest", "balance_dest_after",
            "solde_dest_apres", "final_balance_dest", "after_balance_dest",
            "balance_after_dest", "post_balance_dest", "updated_balance_dest",
        ],
        compound_signals=[
            [
                ["new", "after", "apres", "final", "post", "current", "updated"],
                ["bal", "solde", "balance"],
                ["dest", "to", "recip", "benef"],
            ],
        ],
        forbidden_tokens=["orig", "org", "source", "src", "old", "before", "avant"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Résultat du mapping
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MappingResult:
    df: pd.DataFrame
    mapping: dict[str, str]          # canonical → original_col
    confidence: dict[str, float]     # canonical → score
    warnings: list[str]
    unmapped: list[str]              # champs requis non trouvés

    @property
    def success(self) -> bool:
        return len(self.unmapped) == 0

    def summary(self) -> str:
        lines = ["=== Résultat du mapping sémantique ==="]
        for canonical, original in self.mapping.items():
            score = self.confidence.get(canonical, 0)
            flag = " ⚠" if score < 0.75 else ""
            lines.append(f"  {canonical:20s} ← '{original}' (conf={score:.2f}){flag}")
        if self.unmapped:
            lines.append(f"\nChamps requis non trouvés : {self.unmapped}")
        if self.warnings:
            lines.append("\nAvertissements :")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Réduit une chaîne à ses caractères alphanumériques en minuscules."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _tokenize(s: str) -> list[str]:
    """Découpe en tokens sur _, espace, camelCase, chiffre→lettre."""
    # Découpe camelCase : oldbalanceOrg → ["old", "balance", "Org"]
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    # Découpe sur séparateurs
    parts = re.split(r"[^a-zA-Z0-9]+", s)
    return [p.lower() for p in parts if p]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring d'une colonne pour un champ sémantique
# ─────────────────────────────────────────────────────────────────────────────

def _score_column(col: str, field: SemanticField) -> float:
    """
    Retourne un score de confiance [0, 1] indiquant à quel point `col`
    correspond au champ sémantique `field`.
    """
    col_norm = _normalize(col)
    col_tokens = set(_tokenize(col))

    # ── Niveau 0 : tokens interdits → éliminatoire (comparaison sur les tokens,
    # pas sur la sous-chaîne normalisée — évite que "old" matche dans "solde")
    for tok in field.forbidden_tokens:
        if tok in col_tokens:
            return 0.0

    # ── Niveau 1 : alias exact (original ou normalisé)
    if col.lower() in [a.lower() for a in field.exact_aliases]:
        return 1.0
    if col_norm in [_normalize(a) for a in field.exact_aliases]:
        return 0.95

    # ── Niveau 2 : signaux composés
    # compound_signals est une liste de "clauses" (OU entre clauses).
    # Chaque clause est une liste de groupes de tokens (ET entre groupes).
    # Une clause est satisfaite si, pour chaque groupe, la colonne contient
    # au moins un token du groupe. Le score = 0.80 si la clause la plus
    # exigeante (nb groupes > 1) est satisfaite.
    best_compound = 0.0
    for clause in field.compound_signals:
        non_empty_groups = [g for g in clause if g]
        if not non_empty_groups:
            continue
        satisfied = all(
            any(tok in col_norm for tok in group)
            for group in non_empty_groups
        )
        if satisfied:
            # Récompense les clauses avec plusieurs groupes (plus spécifiques)
            specificity = min(0.80 + 0.04 * (len(non_empty_groups) - 1), 0.90)
            best_compound = max(best_compound, specificity)

    if best_compound > 0:
        return best_compound

    # ── Niveau 3 : fuzzy matching sur les alias normalisés
    aliases_norm = [_normalize(a) for a in field.exact_aliases]
    best_fuzzy = max(
        SequenceMatcher(None, col_norm, alias).ratio()
        for alias in aliases_norm
    )
    if best_fuzzy >= 0.80:
        return round(0.60 * best_fuzzy, 3)

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Moteur de mapping
# ─────────────────────────────────────────────────────────────────────────────

class ColumnMapper:
    """
    Détecte et normalise sémantiquement les colonnes d'un DataFrame de transactions.

    Usage :
        mapper = ColumnMapper()
        result = mapper.map(df)
        if not result.success:
            raise ValueError(result.unmapped)
        df_canonical = result.df
    """

    LOW_CONFIDENCE_THRESHOLD = 0.70

    def map(self, df: pd.DataFrame) -> MappingResult:
        """
        Mappe les colonnes du DataFrame vers les noms canoniques transactionnels.
        Aucune colonne n'est affectée à deux champs distincts.
        """
        columns = list(df.columns)

        # Matrice de scores : {field_name: {col: score}}
        score_matrix: dict[str, dict[str, float]] = {}
        for fname, fdef in SEMANTIC_FIELDS.items():
            score_matrix[fname] = {
                col: _score_column(col, fdef)
                for col in columns
            }

        # Affectation greedy : (score, field, col) trié par score desc
        candidates = sorted(
            [
                (score, fname, col)
                for fname, col_scores in score_matrix.items()
                for col, score in col_scores.items()
                if score > 0
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        assigned_fields: set[str] = set()
        assigned_cols: set[str] = set()
        mapping: dict[str, str] = {}
        confidence: dict[str, float] = {}

        for score, fname, col in candidates:
            if fname in assigned_fields or col in assigned_cols:
                continue
            mapping[fname] = col
            confidence[fname] = score
            assigned_fields.add(fname)
            assigned_cols.add(col)

        # Avertissements pour les mappings incertains
        warnings: list[str] = []
        for fname, score in confidence.items():
            if score < self.LOW_CONFIDENCE_THRESHOLD:
                orig = mapping[fname]
                warnings.append(
                    f"Mapping incertain ({score:.2f}): '{orig}' → '{fname}' "
                    f"({SEMANTIC_FIELDS[fname].description})"
                )

        # Champs requis non trouvés
        unmapped = [
            fname
            for fname, fdef in SEMANTIC_FIELDS.items()
            if fdef.required and fname not in mapping
        ]

        if unmapped:
            # Dernier recours : chercher des colonnes non encore affectées
            # avec n'importe quel score > 0, y compris très bas
            remaining_cols = [c for c in columns if c not in assigned_cols]
            for fname in list(unmapped):
                fdef = SEMANTIC_FIELDS[fname]
                best_col = max(
                    remaining_cols,
                    key=lambda c: _score_column(c, fdef),
                    default=None,
                )
                if best_col is not None:
                    score = _score_column(best_col, fdef)
                    if score > 0.30:
                        mapping[fname] = best_col
                        confidence[fname] = score
                        assigned_cols.add(best_col)
                        unmapped.remove(fname)
                        warnings.append(
                            f"Mapping très incertain ({score:.2f}): "
                            f"'{best_col}' → '{fname}' "
                            f"({fdef.description})"
                        )

        # Construire le DataFrame renommé
        rename_map = {orig: canonical for canonical, orig in mapping.items()}
        df_renamed = df.rename(columns=rename_map)

        # Normaliser les valeurs de la colonne `type` si présente
        if "type" in df_renamed.columns:
            df_renamed = _normalize_type_values(df_renamed)

        return MappingResult(
            df=df_renamed,
            mapping=mapping,
            confidence=confidence,
            warnings=warnings,
            unmapped=unmapped,
        )

    def map_or_raise(self, df: pd.DataFrame) -> MappingResult:
        """Identique à map(), mais lève ValueError si un champ requis est absent."""
        result = self.map(df)
        if not result.success:
            details = _build_missing_error(result, df)
            raise ValueError(details)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation des valeurs de la colonne `type`
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_type_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les valeurs de la colonne `type` vers les constantes transactionnelles
    (TRANSFER, CASH_OUT, CASH_IN, DEBIT, PAYMENT).
    Les valeurs inconnues sont laissées en majuscules.
    """
    def _map_value(v: str) -> str:
        v_norm = str(v).strip().lower().replace(" ", "_")
        return TYPE_VALUE_MAP.get(v_norm, str(v).upper())

    df = df.copy()
    df["type"] = df["type"].apply(_map_value)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Message d'erreur informatif
# ─────────────────────────────────────────────────────────────────────────────

def _build_missing_error(result: MappingResult, df: pd.DataFrame) -> str:
    lines = [
        f"Colonnes requises non trouvées après détection sémantique : {result.unmapped}",
        f"Colonnes présentes dans le CSV : {list(df.columns)}",
        "",
        "Mappings réussis :",
    ]
    for canonical, original in result.mapping.items():
        score = result.confidence.get(canonical, 0)
        lines.append(f"  {canonical} ← '{original}' (conf={score:.2f})")
    lines.append("")
    lines.append(
        "Noms de colonnes acceptés pour les champs manquants :\n"
        + "\n".join(
            f"  {fname}: {', '.join(SEMANTIC_FIELDS[fname].exact_aliases[:8])} ..."
            for fname in result.unmapped
        )
    )
    return "\n".join(lines)


# Singleton
_mapper = ColumnMapper()


def map_columns(df: pd.DataFrame) -> MappingResult:
    """Point d'entrée public : mappe les colonnes d'un DataFrame de transactions."""
    return _mapper.map(df)
