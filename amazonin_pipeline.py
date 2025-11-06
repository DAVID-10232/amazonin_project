# amazonin_pipeline.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import pyarrow.parquet as pq
import pyarrow as pa
import os

# ==============================
# 1️⃣ Ingesta
# ==============================
print("📥 Iniciando ingesta de datos...")
df = pd.read_csv("data/Amazonin1.csv", encoding='utf-8')
df.columns = [c.strip().replace(" ", "_").replace(".", "").lower() for c in df.columns]

# ==============================
# 2️⃣ Limpieza y transformación
# ==============================
print("🧹 Limpiando datos...")
df.dropna(subset=["tweet"], inplace=True)
df["tweet"] = df["tweet"].astype(str).str.lower()

# ==============================
# 3️⃣ Análisis de sentimiento
# ==============================
print("💬 Analizando sentimiento...")
analyzer = SentimentIntensityAnalyzer()
df["compound"] = df["tweet"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment"] = df["compound"].apply(
    lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
)

# ==============================
# 4️⃣ Almacenamiento optimizado
# ==============================
print("💾 Guardando resultados optimizados...")
os.makedirs("output", exist_ok=True)
table = pa.Table.from_pandas(df)
pq.write_table(table, "output/resultados.parquet")

# ==============================
# 5️⃣ Informe resumen (PDF)
# ==============================
print("📊 Generando informe PDF...")
summary = df["sentiment"].value_counts().to_dict()

pdf_path = "output/informe_resumen.pdf"
c = canvas.Canvas(pdf_path, pagesize=A4)
c.setFont("Helvetica-Bold", 16)
c.drawString(50, 800, "Informe de Sentimiento - AMAZONIN Tweets")
c.setFont("Helvetica", 12)
c.drawString(50, 770, f"Tweets analizados: {len(df)}")
y = 740
for k, v in summary.items():
    c.drawString(50, y, f"{k.capitalize()}: {v}")
    y -= 20
c.save()

print("✅ Pipeline completado con éxito.")
print(f"Resultados en: output/resultados.parquet y output/informe_resumen.pdf")
