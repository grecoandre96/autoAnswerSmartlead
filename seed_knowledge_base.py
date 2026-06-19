"""
Legge dataset_addestramento.jsonl, campiona esempi per categoria,
li generalizza con Gemini, e li carica nel database SQLite.
Da eseguire una volta (o ogni volta che si vuole aggiornare la knowledge base).
"""

import json
import sqlite3
import os
import sys
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

DB_PATH = Path(__file__).parent / "qa_data.db"
DATASET_PATH = Path(__file__).parent.parent / "dataset_addestramento.jsonl"

# Categorie che non richiedono risposta — le saltiamo o aggiungiamo una regola fissa
SKIP_CATEGORIES = {4, 6}  # OUT_OF_OFFICE

# Quanti esempi campionare per categoria per farli generalizzare a Gemini
SAMPLES_PER_CATEGORY = 4

CATEGORY_LABELS = {
    1: "INTERESTED — interessato, vuole procedere o approfondire",
    2: "BOOKING_READY — propone date specifiche per un incontro",
    3: "INFO_REQUEST — chiede presentazione o informazioni generali",
    4: "OUT_OF_OFFICE — risposta automatica di assenza",
    5: "INFO_REQUEST_DETAILED — chiede chiarimenti specifici sul servizio",
    6: "OUT_OF_OFFICE — risposta automatica di assenza (altra variante)",
    7: "REFERRAL — gira la mail a un collega fornendo nome/email",
    8: "MEETING_BOOKED — conferma o propone slot preciso",
    9: "EMAIL_REDIRECT — casella non attiva, rimanda ad altro indirizzo",
    75712: "INFO_REQUEST — chiede presentazione o profilo aziendale",
    77331: "INTERESTED_CURIOUS — interessato ma con domande o richieste di chiarimento",
    None: "UNCATEGORIZED — risposta varia non categorizzata",
}

GENERALIZE_SYSTEM_PROMPT = """Sei un esperto di sales automation. Ti vengono forniti esempi reali di email di lead B2B e le relative risposte degli agenti di vendita.

CONTESTO IMPORTANTE: Le email analizzate sono RISPOSTE di lead a nostre email di cold outreach — siamo stati NOI a contattarli per primi. Tienilo presente nel descrivere la situazione e nel formulare la risposta ideale.

Il tuo compito è estrarre da questi esempi UN SINGOLO pattern generico e riutilizzabile:
1. SITUAZIONE: descrivi il tipo di risposta del lead (senza nomi, aziende, link o dati specifici)
2. RISPOSTA IDEALE: la risposta che un agente di vendita dovrebbe dare — senza "grazie per averci contattato" (sono loro che rispondono a noi), senza link specifici, senza nomi aziendali. Usa placeholder come [LINK CALENDARIO], [LINK PRESENTAZIONE], [NOME AGENTE] dove necessario.

La risposta deve essere abbastanza generica da applicarsi a qualsiasi settore B2B, ma abbastanza concreta da essere una guida utile per un AI sales assistant.

Rispondi SOLO con un JSON valido senza markdown:
{
  "context": "descrizione della situazione tipica del lead (2-3 frasi)",
  "response": "risposta ideale generalizzata pronta all'uso, max 120 parole, con saluto e firma generici"
}"""


def load_dataset():
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def group_by_category(records):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r.get("lead_category_id")].append(r)
    return groups


def generalize_examples(category_id, category_label, examples):
    """Passa esempi a Gemini e ottieni un pattern generico."""
    examples_text = "\n\n---\n\n".join(
        f"EMAIL LEAD:\n{ex['messaggio_cliente'][:600]}\n\nRISPOSTA AGENTE:\n{ex['risposta_agente'][:600]}"
        for ex in examples
    )

    prompt = f"""Categoria: {category_label}

Ecco {len(examples)} esempi reali di questa situazione:

{examples_text}

Estrai il pattern generico per questa categoria."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=GENERALIZE_SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_qa(conn, context, response):
    conn.execute(
        "INSERT INTO qa_pairs (context, response) VALUES (?, ?)",
        (context, response),
    )
    conn.commit()


def main():
    print("=== Seed Knowledge Base ===\n")

    if not DATASET_PATH.exists():
        print(f"ERRORE: dataset non trovato in {DATASET_PATH}")
        sys.exit(1)

    print(f"Carico dataset da {DATASET_PATH}...")
    records = load_dataset()
    print(f"Totale record: {len(records)}")

    groups = group_by_category(records)
    conn = init_db()

    # Chiedi se svuotare prima il DB
    existing = conn.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0]
    if existing > 0:
        print(f"\nATTENZIONE: il database ha già {existing} esempi.")
        answer = input("Vuoi svuotarlo prima di aggiungere i nuovi? (s/n): ").strip().lower()
        if answer == "s":
            conn.execute("DELETE FROM qa_pairs")
            conn.commit()
            print("Database svuotato.\n")

    added = 0
    for cat_id, cat_records in groups.items():
        if cat_id in SKIP_CATEGORIES:
            print(f"[SKIP] cat={cat_id} (OUT_OF_OFFICE — nessuna risposta necessaria)")
            continue

        label = CATEGORY_LABELS.get(cat_id, f"UNKNOWN (id={cat_id})")
        print(f"\n[cat={cat_id}] {label} — {len(cat_records)} record")

        # Campiona esempi
        sample = random.sample(cat_records, min(SAMPLES_PER_CATEGORY, len(cat_records)))

        try:
            result = generalize_examples(cat_id, label, sample)
            context = result.get("context", "").strip()
            response = result.get("response", "").strip()

            if context and response:
                save_qa(conn, context, response)
                added += 1
                print(f"  OK: '{context[:80]}...'")
            else:
                print(f"  WARN: risposta Gemini vuota o malformata")
        except Exception as e:
            print(f"  ERRORE: {e}")

    conn.close()
    print(f"\n=== Completato: {added} esempi aggiunti al database ===")
    print(f"Avvia l'app e vai su http://localhost:5000 per vederli nella dashboard.")


if __name__ == "__main__":
    main()
