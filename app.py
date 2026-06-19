import os
import json
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
_gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

app = Flask(__name__)

# Il DB vive su un volume persistente se DB_DIR è configurato (es. /data su Railway),
# altrimenti nella cartella dell'app (sviluppo locale).
DB_DIR = os.environ.get("DB_DIR", os.path.dirname(__file__))
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "qa_data.db")

# Default system prompt — usato solo al primo avvio per popolare il DB.
# Dopo il primo deploy viene letto e modificato dal DB via dashboard.
SYSTEM_PROMPT = """[CONTESTO — LEGGI CON ATTENZIONE]
Sei un AI Sales Assistant. Le email che analizzi sono RISPOSTE di lead B2B a una nostra email di cold outreach: siamo stati NOI a contattarli per primi. Il lead non ci ha cercati — ha risposto a una nostra proposta.

Questo significa:
- NON usare mai frasi come "grazie per averci contattato" o "grazie per averci scritto" — siamo noi che abbiamo scritto a loro
- Il tono corretto è: professionale, diretto, senza essere servile
- Frasi come "grazie per il riscontro" o "grazie per la risposta" sono accettabili perché ringraziano per aver risposto alla nostra email
- Ogni risposta deve portare il lead al passo successivo (call, meeting, info utile) senza pressione

[REGOLE INDEROGABILI]
- Rispondi SEMPRE nella lingua usata dal lead nell'email (se scrive in inglese → inglese, francese → francese, ecc.)
- Non rivelare mai di essere un AI
- Usa il nome del lead se disponibile, altrimenti ometti il saluto nominale
- Sii diretto, niente frasi di riempimento ("Spero che lei stia bene", "Come sta?")
- Se c'è un link Calendly, usalo per proporre la call — non chiedere disponibilità generica
- Non superare 120 parole nel corpo email
- Non usare emoji
- Firma sempre con: Nome + Ruolo + Azienda (come nel profilo aziendale)
- Sequenza 1 = tono più formale; sequenza 2+ = più diretto e familiare

[KNOWLEDGE BASE — SITUAZIONI TIPICHE E RISPOSTE]
Esempi reali di situazioni ricorrenti. Adatta la risposta al contesto specifico — non copiare alla lettera, ma usa questi esempi come guida di stile e contenuto:

{qa_examples}

[CATEGORIE DI INTENT]
Categorizza l'email del lead in UNO di questi intent:

- BOOKING_READY: propone o conferma date/orari per incontro o call
- INFO_REQUEST: chiede presentazione, info sul servizio, prezzi, modalità
- REFERRAL: gira la mail a un collega e fornisce nome/email del nuovo referente
- OBJECTION_SEASONAL: rimanda citando stagionalità, periodo sbagliato, fine anno, ecc.
- RESCHEDULE: vuole essere ricontattato più avanti senza data precisa
- REJECTION: rifiuto netto ("non siamo interessati", "abbiamo già qualcuno", ecc.)
- OUT_OF_OFFICE: risposta automatica di assenza o casella non presidiata
- ESCALATE: domande tecniche, legali o molto specifiche che richiedono risposta umana
- OTHER: qualsiasi situazione non rientrante nelle precedenti

Azioni per intent speciali:
- REJECTION → draft_email vuota, non rispondere
- OUT_OF_OFFICE → draft_email vuota, non rispondere
- ESCALATE → draft_email vuota, requires_human: true

[FORMATO DELLA BOZZA EMAIL]
La draft_email deve essere una email COMPLETA e pronta al copia-incolla, con questa struttura:

Gentile [Nome],        ← saluto formale (o "Ciao [Nome]," se tono informale da sequenza 2+)

[corpo email — max 120 parole]

%firma%

Usa "\n\n" (doppio a capo) tra i paragrafi — mai "\n" singolo tra paragrafi distinti. Termina SEMPRE con "%firma%" — niente nomi, ruoli o aziende scritti a mano. La email deve essere immediatamente inviabile senza modifiche, salvo inserire eventuali link mancanti indicati come [LINK CALENDARIO] o [LINK PRESENTAZIONE].

[FORMATO DI OUTPUT JSON]
Restituisci ESCLUSIVAMENTE un oggetto JSON valido, senza blocchi markdown:
{
  "intent": "string",
  "reasoning": "string (1-2 frasi sul perché di quell'intent)",
  "requires_human": boolean,
  "extracted_data": {
    "suggested_dates": "string o null",
    "new_contact_email": "string o null",
    "new_contact_name": "string o null"
  },
  "draft_email": "string (email completa formattata, vuota se REJECTION / OUT_OF_OFFICE / ESCALATE)"
}"""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()

    # Se il DB è vuoto, carica gli esempi dal file seed
    count = conn.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0]
    if count == 0:
        seed_path = os.path.join(os.path.dirname(__file__), "seed_data.json")
        if os.path.exists(seed_path):
            with open(seed_path, "r", encoding="utf-8") as f:
                examples = json.load(f)
            for ex in examples:
                conn.execute(
                    "INSERT INTO qa_pairs (context, response) VALUES (?, ?)",
                    (ex["context"], ex["response"]),
                )
            conn.commit()

    # Popola il system prompt di default se non esiste ancora
    existing = conn.execute("SELECT 1 FROM settings WHERE key='system_prompt'").fetchone()
    if not existing:
        conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)", ("system_prompt", SYSTEM_PROMPT))
        conn.commit()

    conn.close()


def build_system_prompt():
    conn = get_db()
    prompt_row = conn.execute("SELECT value FROM settings WHERE key='system_prompt'").fetchone()
    base_prompt = prompt_row["value"] if prompt_row else SYSTEM_PROMPT
    rows = conn.execute("SELECT context, response FROM qa_pairs ORDER BY id").fetchall()
    conn.close()

    if not rows:
        qa_section = "(Nessun esempio ancora caricato — aggiungili dalla dashboard)"
    else:
        examples = []
        for i, row in enumerate(rows, 1):
            examples.append(
                f'SITUAZIONE {i}: "{row["context"]}"\n'
                f'RISPOSTA SUGGERITA: "{row["response"]}"'
            )
        qa_section = "\n\n".join(examples)

    return base_prompt.replace("{qa_examples}", qa_section)


# ---------------------------------------------------------------------------
# API endpoint chiamato da n8n
# ---------------------------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze_email():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Il corpo della richiesta deve essere JSON"}), 400

        email_cliente = data.get("email_cliente", "").strip()
        profilo_aziendale = data.get("profilo_aziendale", "").strip()

        if not email_cliente or not profilo_aziendale:
            return jsonify({"error": "Campi obbligatori mancanti: email_cliente, profilo_aziendale"}), 400

        lead_first_name = data.get("lead_first_name", "")
        lead_company = data.get("lead_company", "")
        sequence_number = data.get("sequence_number", "")
        calendly_link = data.get("calendly_link", "")

        context_lines = [
            f"PROFILO AZIENDALE:\n{profilo_aziendale}",
        ]
        if calendly_link:
            context_lines.append(f"Link calendario (Calendly): {calendly_link}")
        context_lines.append(
            f"\nDATI LEAD:\n"
            f"- Nome: {lead_first_name}\n"
            f"- Azienda: {lead_company or 'non specificata'}\n"
            f"- Sequenza campagna: {sequence_number or 'non specificata'}"
        )
        context_lines.append(f"\nEMAIL DEL LEAD:\n{email_cliente}")

        prompt = "\n".join(context_lines)

        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(),
                response_mime_type="application/json",
            ),
        )
        result = json.loads(response.text)
        return jsonify(result), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Impossibile fare il parsing della risposta come JSON"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Dashboard routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    conn = get_db()
    qa_pairs = conn.execute("SELECT * FROM qa_pairs ORDER BY id DESC").fetchall()
    prompt_row = conn.execute("SELECT value FROM settings WHERE key='system_prompt'").fetchone()
    current_prompt = prompt_row["value"] if prompt_row else SYSTEM_PROMPT
    conn.close()
    return render_template("dashboard.html", qa_pairs=qa_pairs, system_prompt=current_prompt)


@app.route("/settings/system-prompt", methods=["POST"])
def update_system_prompt():
    new_prompt = request.form.get("system_prompt", "").strip()
    if new_prompt:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            ("system_prompt", new_prompt),
        )
        conn.commit()
        conn.close()
    return redirect(url_for("dashboard"))


@app.route("/qa/add", methods=["POST"])
def add_qa():
    context = request.form.get("context", "").strip()
    response_text = request.form.get("response", "").strip()
    if context and response_text:
        conn = get_db()
        conn.execute("INSERT INTO qa_pairs (context, response) VALUES (?, ?)", (context, response_text))
        conn.commit()
        conn.close()
    return redirect(url_for("dashboard"))


@app.route("/qa/edit/<int:qa_id>", methods=["POST"])
def edit_qa(qa_id):
    context = request.form.get("context", "").strip()
    response_text = request.form.get("response", "").strip()
    if context and response_text:
        conn = get_db()
        conn.execute(
            "UPDATE qa_pairs SET context=?, response=? WHERE id=?",
            (context, response_text, qa_id),
        )
        conn.commit()
        conn.close()
    return redirect(url_for("dashboard"))


@app.route("/qa/delete/<int:qa_id>", methods=["POST"])
def delete_qa(qa_id):
    conn = get_db()
    conn.execute("DELETE FROM qa_pairs WHERE id=?", (qa_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("dashboard"))


init_db()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
