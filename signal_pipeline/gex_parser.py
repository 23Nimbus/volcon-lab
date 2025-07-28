GEX_TRIGGERS = {
    # terms describing potential support zones or buy walls
    "support_terms": ["support", "dip zone", "buy wall"],
    # terms describing potential resistance zones or sell walls
    "resistance_terms": ["resistance", "rip zone", "sell wall"],
    "gamma_break_terms": ["gamma break", "snap zone", "vol zone"],
    "cluster_weakness_terms": ["weak cluster", "fragile", "thin gamma"],
    "macro_overlay_terms": ["FOMC", "tariff", "earnings", "CPI"]
}


def parse_gex_comment(text: str):
    text = text.lower()
    return {
        "support_discussed": any(
            term.lower() in text for term in GEX_TRIGGERS["support_terms"]
        ),
        "resistance_discussed": any(
            term.lower() in text for term in GEX_TRIGGERS["resistance_terms"]
        ),
        "gamma_break_near": any(term.lower() in text for term in GEX_TRIGGERS["gamma_break_terms"]),
        "fragile_containment": any(term.lower() in text for term in GEX_TRIGGERS["cluster_weakness_terms"]),
        "macro_risk_overlay": any(term.lower() in text for term in GEX_TRIGGERS["macro_overlay_terms"])
    }
