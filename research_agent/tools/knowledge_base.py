"""
Mock knowledge base — a small corpus standing in for "past reports / filings
/ analyst notes" that a real RAG system would index from a document store
(internal Google Drive, a vector DB of past research reports, SEC filings,
earnings call transcripts, etc.).

MOCKED for this sandbox in the sense that the *content* is invented, but
the retrieval mechanism in rag_tools.py that runs over this data is real
TF-IDF + cosine similarity, not a stub.

TO GO LIVE:
    - replace DOCUMENTS with real ingested content (chunked filings,
      past reports, transcripts)
    - for larger corpora, swap the TF-IDF retriever in rag_tools.py for a
      real embedding-based vector store (e.g. Chroma/FAISS + an embedding
      API) — see rag_tools.py docstring for the exact swap point.
"""

DOCUMENTS = [
    {
        "id": "doc1",
        "title": "AAPL — Historical volatility and earnings reaction patterns",
        "tags": ["AAPL"],
        "text": (
            "Apple shares have historically shown muted single-day "
            "volatility around earnings compared to peers, typically "
            "moving 2-4% versus 6-10% for smaller-cap tech names. Product "
            "launch events tend to have a smaller market impact than "
            "earnings surprises. Services segment growth has been the "
            "primary driver of multiple expansion over the past several "
            "years, more so than iPhone unit growth alone."
        ),
    },
    {
        "id": "doc2",
        "title": "TSLA — Production ramp and delivery guidance track record",
        "tags": ["TSLA"],
        "text": (
            "Tesla's stock has historically reacted more strongly to "
            "delivery numbers and production ramp commentary than to "
            "traditional GAAP earnings beats or misses. Guidance changes "
            "on annual delivery targets have historically produced larger "
            "single-day moves than the earnings report itself. Margin "
            "commentary on new factories has been a recurring point of "
            "analyst focus."
        ),
    },
    {
        "id": "doc3",
        "title": "NVDA — AI accelerator market position and competitive notes",
        "tags": ["NVDA"],
        "text": (
            "Nvidia's data center segment has been the primary growth "
            "driver, with GPU demand tied closely to AI training and "
            "inference workloads. Competitive pressure from custom silicon "
            "(in-house chips built by large cloud providers) is a "
            "recurring theme in analyst notes, though production capacity "
            "constraints have historically mattered more than competition "
            "for near-term results."
        ),
    },
    {
        "id": "doc4",
        "title": "General — How to read a 10-K risk factors section",
        "tags": [],
        "text": (
            "When reviewing a 10-K's risk factors section, look for "
            "language that changed from the prior year's filing rather "
            "than boilerplate that repeats annually. New risk factors, or "
            "reordering that moves a risk higher in the list, often "
            "signals where management's own concern has shifted, even "
            "when the language itself sounds generic."
        ),
    },
    {
        "id": "doc5",
        "title": "General — Interpreting analyst price target revisions",
        "tags": [],
        "text": (
            "A cluster of price target revisions in the same direction "
            "shortly after an earnings report is typically a lagging "
            "signal, not a leading one — it reflects analysts updating "
            "models after new guidance rather than predicting it. The "
            "size of the revision relative to the stock's historical "
            "volatility matters more than the raw dollar change."
        ),
    },
    {
        "id": "doc6",
        "title": "AMZN — AWS margin trends and retail segment context",
        "tags": ["AMZN"],
        "text": (
            "Amazon's overall margin profile has become increasingly "
            "dependent on AWS performance, since the retail segment "
            "operates on much thinner margins. Analyst focus in recent "
            "periods has shifted toward AWS growth deceleration/"
            "reacceleration as the key swing factor for the stock, more "
            "than retail unit growth."
        ),
    },
    {
        "id": "doc7",
        "title": "General — Sector rotation and mega-cap tech correlation",
        "tags": [],
        "text": (
            "Mega-cap technology stocks have shown elevated correlation "
            "with each other during broad risk-off market periods, "
            "meaning company-specific news can be temporarily overwhelmed "
            "by macro sentiment. Distinguishing a stock-specific move from "
            "a sector-wide move is important before over-interpreting a "
            "single day's price action."
        ),
    },
    {
        "id": "doc8",
        "title": "MSFT — Cloud segment reporting and Copilot monetization",
        "tags": ["MSFT"],
        "text": (
            "Microsoft's Azure growth rate is the most closely watched "
            "single metric in its quarterly reports. Commentary on AI "
            "Copilot monetization and enterprise seat attach rates has "
            "become an increasingly common focus area in analyst "
            "questions during earnings calls."
        ),
    },
]
