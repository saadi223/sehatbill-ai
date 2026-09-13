**SehatBill AI — Document Q&A**

Upload selectable-text PDF or UTF-8 TXT files, then ask questions about them. The app extracts PDF text page by page with PyMuPDF, splits each page into overlapping chunks, embeds them locally with sentence-transformers/all-MiniLM-L6-v2, and searches a per-session FAISS index. It sends your question and the most relevant excerpts to Groq's openai/gpt-oss-20b to write a brief answer with [S1]-style citations. Each cited filename, page and excerpt appears below the answer. A TXT file is labeled page 1 because it has no physical pages.

The interface walks users through upload, question, and source checking. After upload, it shows clickable question ideas. General ideas always appear; bill, policy, and provider questions appear when related words are found in the document. Suggestions are templates, not confirmed facts. Clicking one asks it immediately. You can also type a question. Simple English retrieval works best; Roman Urdu may miss relevant text with this English-focused embedding model. The last answer stays on screen until the files change or are removed.

This is a learning demo for fictional or non-sensitive documents, not medical or financial advice. It does not OCR scans, diagnose conditions, verify that charges are fraudulent, or determine whether a price is fair without an authoritative tariff. Retrieval and generated answers can still be wrong; verify against the displayed excerpts. The similarity cutoff is a heuristic, so even an answer present in a document may sometimes be missed. Excerpts and questions leave Streamlit's server for Groq. Each browser session holds its own index in memory; uploads are not written to disk by this app. A new or restarted session must upload again. A public app should not be used for real patient data without a suitable privacy and security review.

**Deploy using websites only**

Create a GitHub account and a Groq API key. Keep the key private.

On GitHub, create a repository named sehatbill-ai. Choose Private if you prefer, subject to your Streamlit account's repository access. Open the repository, choose Add file → Create new file, enter app.py, paste the supplied file contents, and click Commit changes. Repeat for requirements.txt, README.md, .gitignore, and .streamlit/config.toml (type the slash in the filename field to create the folder). Never upload a secrets file or real medical bills to the repository.

Open Streamlit Community Cloud, sign in with GitHub, and choose Create app → Deploy a public app from GitHub (wording may vary). Select your repository and branch, set main file path to app.py, and give the app a URL.

In Advanced settings during deployment, paste this TOML into Secrets (replace the placeholder with your actual key):

GROQ_API_KEY = "your_actual_groq_key"

Save and deploy. If you already deployed, open the app's Manage app / Settings → Secrets, enter the same TOML, and save or restart as prompted. The key belongs only in Streamlit Secrets; do not put it in GitHub or a URL.

Wait while Streamlit installs requirements.txt and downloads the embedding model on its first run. Upload a small, selectable-text PDF or UTF-8 TXT and ask a question whose answer appears in it. Try an unrelated question to see the unsupported-answer behavior. For deploy errors, open Manage app → Logs and check the requirements, secrets, and main file path.

Upload limits in this demo: 10 MB per file, 25 MB total, 150 pages per PDF, 500,000 extracted characters per document. A scanned PDF with no text needs OCR outside this app. On a small free instance, model loading or many uploads may exhaust memory; try smaller files. faiss-cpu wheels depend on the deployment Python version; use a supported Python version in Streamlit's Advanced settings if package installation fails.
The .streamlit/config.toml file sets Streamlit's upload-widget limit to 25 MB so it no longer advertises the default 200 MB; the app separately enforces 10 MB per file and 25 MB combined.

**Files**

app.py: user interface, extraction, retrieval, generation and source display.

requirements.txt: Python packages Streamlit installs automatically.

.gitignore: excludes secrets, local environments and sample PDFs/TXT files.

.streamlit/config.toml: matches the upload widget limit to the app's total upload limit.

No Colab, VS Code, or terminal is needed for deployment.
