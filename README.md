# Arbitration Rejoinder Drafter

A Streamlit app that drafts a Markdown Rejoinder on behalf of the Claimant from:

- a required Statement of Claim;
- a required Statement of Defence; and
- optional supporting documents.

Supported uploads are PDF, DOCX, TXT, and Markdown. PDFs must contain selectable text; scanned PDFs need OCR before upload.

## Setup

Create a `.env` file (or use the existing one):

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model_id
```

Then install and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app streams the draft to the page, provides a Markdown editing field, and downloads the result as `claimant_rejoinder.md`.

Relevant supporting documents that were not already exhibited with the Statement of Claim are introduced as new Claimant exhibits. Existing exhibit references are reused; where the exhibit sequence cannot be determined safely, the draft inserts a counsel placeholder. A schedule of newly introduced exhibits is appended to the Rejoinder.

## Important

Uploaded document text is sent to the model configured in `.env`. Review confidentiality obligations and provider data-handling terms before using client materials. The generated draft must be reviewed by qualified counsel before filing or reliance.
