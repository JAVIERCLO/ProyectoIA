import streamlit as st
import os, base64, json, re
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Dict, Any, Optional
import pandas as pd

import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="FarmacIA Asistente para medicinas.", layout="wide")

st.markdown("""
# FarmacIA - Asistente de recetas médicas
**Aviso importante:** Esta aplicación **no sustituye** ni supera el consejo médico profesional. 
No cambia medicamentos recetados. Solo ayuda a **interpretar** la receta, **planificar** tomas y **mostrar** presentaciones equivalentes de **la misma sustancia**.
Si tienes dudas, **consulta a tu médico o farmacéutico**.
""")

def encode_image_to_data_url(file_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def safe_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def parse_mg(text: str) -> Optional[float]:
    if not text: 
        return None
    t = safe_lower(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg", t)
    return float(m.group(1)) if m else None

def parse_ml(text: str) -> Optional[float]:
    if not text: 
        return None
    t = safe_lower(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*ml", t)
    return float(m.group(1)) if m else None

def human_qty_label(qty: float, unit: str) -> str:
    if unit == "ml":
        return f"{qty:.1f} ml"
    return f"{qty:.2f} {unit}"


SYSTEM_JSON = (
    "Eres un asistente experto en extracción de información de recetas médicas en español. "
    "Devuelve SOLO JSON válido con la estructura:\n"
    "{"
    "  'paciente': {'nombre': str | null}, "
    "  'doctor': {'nombre': str | null, 'colegiado': str | null}, "
    "  'prescripciones': ["
    "     {'medicamento': str, 'dosis': str, 'via': str | null, 'frecuencia': str, 'duracion_dias': int | null, 'instrucciones': str | null}"
    "  ]"
    "} "
    "No inventes datos. Si faltan, usa null."
)

USER_JSON = "Extrae la información en el JSON indicado. Solo JSON, sin texto adicional."

SYSTEM_BOX = (
    "Eres un extractor de etiquetas de medicamentos. Devuelve SOLO JSON con:\n"
    "{"
    " 'active_ingredient': str | null, "
    " 'brand': str | null, "
    " 'form': 'tablet'|'capsula'|'suspension'|'jarabe'|'gota'|'parche'|null, "
    " 'strength_mg_per_unit': number | null, "
    " 'concentration_mg_per_ml': number | null, "
    " 'units_per_pack': number | null, "
    " 'volume_ml': number | null "
    "} "
    "No inventes. Si no se ve, usa null."
)

SYSTEM_EXPLAIN = (
    "Eres un asistente que explica en lenguaje simple los usos comunes (síntomas/enfermedades) del principio activo indicado. "
    "No des dosis ni indicaciones clínicas. No sustituyas consejo médico. Devuelve 3-6 usos frecuentes y 2-3 precauciones genéricas."
)

def gpt_vision_json(image_data_urls: List[str], system_prompt: str, user_text: str) -> Dict[str, Any]:
    content = [{"type":"text","text": user_text}]
    for url in image_data_urls:
        content.append({"type":"image_url","image_url":{"url": url}})
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content": system_prompt},
                {"role":"user","content": content}
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        st.error(f"Error de extracción: {e}")
        return {}

def gpt_text_explain(active_ingredient: str) -> Dict[str, Any]:
    if "explain_cache" not in st.session_state:
        st.session_state["explain_cache"] = {}
    key = safe_lower(active_ingredient)
    if key in st.session_state["explain_cache"]:
        return st.session_state["explain_cache"][key]
    try:
        prompt = (
            f"Principio activo: {active_ingredient}\n"
            "Devuelve SOLO JSON con:\n"
            "{ 'usos_comunes': [str,...], 'precauciones_generales': [str,...] }"
        )
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content": SYSTEM_EXPLAIN},
                {"role":"user","content": prompt}
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        st.session_state["explain_cache"][key] = data
        return data
    except Exception as e:
        st.warning(f"No se pudo generar explicación automática: {e}")
        return {"usos_comunes": [], "precauciones_generales": []}


# Catalogo (CSV)
CATALOG_COLUMNS = [
    "id","active_ingredient","brand","form",
    "strength_mg_per_unit","concentration_mg_per_ml",
    "units_per_pack","unit","volume_ml","presentation_text"
]

def ensure_catalog_template(path="catalogo_meds.csv"):
    if not os.path.exists(path):
        import pandas as pd
        df = pd.DataFrame([{
            "id": 1, "active_ingredient": "ibuprofeno", "brand": "Generico", "form": "tablet",
            "strength_mg_per_unit": 400, "concentration_mg_per_ml": None,
            "units_per_pack": 10, "unit": "tableta", "volume_ml": None,
            "presentation_text": "Tabletas 400 mg (10u)"
        },{
            "id": 2, "active_ingredient": "ibuprofeno", "brand": "Generico", "form": "suspension",
            "strength_mg_per_unit": None, "concentration_mg_per_ml": 20,
            "units_per_pack": None, "unit": "ml", "volume_ml": 120,
            "presentation_text": "Suspensión 20 mg/ml (120 ml)"
        },{
            "id": 3, "active_ingredient": "paracetamol", "brand": "Generico", "form": "tablet",
            "strength_mg_per_unit": 500, "concentration_mg_per_ml": None,
            "units_per_pack": 20, "unit": "tableta", "volume_ml": None,
            "presentation_text": "Tabletas 500 mg (20u)"
        }], columns=CATALOG_COLUMNS)
        df.to_csv(path, index=False)

def load_catalog(path="catalogo_meds.csv") -> 'pd.DataFrame':
    ensure_catalog_template(path)
    import pandas as pd
    try:
        df = pd.read_csv(path)
        for c in ["active_ingredient","brand","form","unit","presentation_text"]:
            if c in df.columns:
                df[c] = df[c].fillna("")
        return df
    except Exception as e:
        st.error(f"Error cargando catálogo: {e}")
        import pandas as pd
        return pd.DataFrame(columns=CATALOG_COLUMNS)

def unidades_por_toma(dosis_mg: float, row: Dict[str, Any]):
    form = safe_lower(str(row.get("form","")))
    strength = row.get("strength_mg_per_unit")
    conc = row.get("concentration_mg_per_ml")
    import pandas as pd
    if form in ("tablet","capsula") and pd.notna(strength):
        mg_unit = float(strength)
        if mg_unit > 0:
            qty = round(dosis_mg / mg_unit, 2)
            return qty, ("tabletas" if form=="tablet" else "capsulas"), f"{qty:.2f} unidades"
    if form in ("jarabe","suspension","gota") and pd.notna(conc):
        mg_ml = float(conc)
        if mg_ml > 0:
            ml = round(dosis_mg / mg_ml, 1)
            return ml, "ml", f"{ml:.1f} ml"
    return None, None, "No se pudo calcular con esta presentación"

def alternativas_misma_sustancia(df, active_ingredient: str, dosis_mg: float):
    alts = []
    if df is None or df.empty: 
        return alts
    for _, row in df.iterrows():
        if safe_lower(row.get("active_ingredient","")) == safe_lower(active_ingredient):
            qty, unidad, nota = unidades_por_toma(dosis_mg, row)
            if qty and unidad:
                alts.append({
                    "brand": row.get("brand",""),
                    "presentation": row.get("presentation_text",""),
                    "por_toma": human_qty_label(qty, "ml" if unidad=="ml" else unidad),
                    "unidad_cruda": unidad,
                    "nota": nota
                })
    return alts

# Frecuencia a horarios de toma
def parse_frequency_to_hours(freq_text: str):
    if not freq_text: return [9,21]
    t = safe_lower(freq_text)
    m = re.search(r"cada\s+(\d{1,2})\s*h", t)
    if m:
        h = int(m.group(1))
        if h <= 0 or h > 24: return [9,21]
        return list(range(0,24,h))
    m = re.search(r"(\d{1,2})\s*veces\s+al\s+d[ií]a", t)
    if m:
        n = int(m.group(1))
        if n <= 0: return [9,21]
        base = 8
        step = 24 / n
        return [int((base + i*step) % 24) for i in range(n)]
    m = re.search(r"cada\s+(\d{1,2})\s*d[ií]as", t)
    if m:
        return [9]
    return [9,21]

def build_events(presc: Dict[str,Any], start_dt: datetime, default_days: int = 5):
    freq = presc.get("frecuencia") or ""
    hours = parse_frequency_to_hours(freq)
    dur = presc.get("duracion_dias") or default_days
    med = presc.get("medicamento","(medicamento)")
    dosis = presc.get("dosis","" )
    events = []
    for d in range(dur):
        for h in hours:
            dt = (start_dt + timedelta(days=d)).replace(hour=h, minute=0, second=0, microsecond=0)
            events.append({"datetime": dt.isoformat(), "label": f"{med} — {dosis}"})
    return events

def csv_printable(rows):
    import csv
    out = StringIO()
    w = csv.writer(out)
    w.writerow(["Fecha/Hora","Medicamento","Dosis por toma","Síntoma/Enfermedad","Notas"])
    for r in rows:
        w.writerow([r["fecha_hora"], r["medicamento"], r["dosis"], r["sintoma_enfermedad"], r.get("notas","")])
    return out.getvalue()

# Advertencias de dosis altas
def dosage_warning(unidad: str, qty: float):
    u = safe_lower(unidad or "")
    if u in ("tabletas","capsulas") and qty is not None and qty > 2:
        return f"Cantidad alta por toma: {qty} {u}. Verifica con tu médico/farmacéutico."
    if u == "ml" and qty is not None and qty > 30:
        return f"Volumen alto por toma: {qty} ml. Verifica con tu médico/farmacéutico."
    return None


# Pestañas
tab1, tab2, tab3 = st.tabs(["1) Procesar receta", "2) Consulta de medicamento (foto o nombre)", "3) Catálogo y alternativas"])


# Pestaña 1: Procesar receta
with tab1:
    st.subheader("Sube imagen de la receta")
    uploaded = st.file_uploader("Receta (JPG/PNG)", type=["png","jpg","jpeg"], key="rx_upload")

    pages = []
    if uploaded is not None:
        mime = uploaded.type
        if mime.startswith("image/"):
            st.image(uploaded, caption="Imagen subida", width="stretch")
            pages.append(encode_image_to_data_url(uploaded.read(), mime))

    if pages:
        with st.spinner("Extrayendo campos (GPT-4o visión)..."):
            data = gpt_vision_json(pages, SYSTEM_JSON, USER_JSON)

        if data:
            st.success("Extracción completada")
            st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
            presc = data.get("prescripciones") or []

            col1, col2 = st.columns(2)

            with col1:
                fecha = st.date_input("Fecha de inicio", value=datetime.now().date())

            with col2:
                hora = st.time_input("Hora de inicio", value=datetime.now().time())

            start = datetime.combine(fecha, hora)

            default_days = st.number_input("Duración por defecto (días) si falta", 1, 60, 5, 1, key="default_days_rx")

            cat = load_catalog()

            all_events = []
            printable_rows = []
            warnings = []

            for i, p in enumerate(presc, start=1):
                st.markdown(f"### Prescripción #{i}")
                colA, colB = st.columns(2)
                with colA:
                    med_name = st.text_input("Medicamento (editable)", value=p.get("medicamento",""), key=f"med_{i}")
                    dosis_text = st.text_input("Dosis (editable, ej. '400 mg')", value=p.get("dosis",""), key=f"dose_{i}")
                    frecuencia = st.text_input("Frecuencia", value=p.get("frecuencia",""), key=f"freq_{i}")
                    duracion = st.number_input("Duración (días)", 1, 60, value=int(p.get("duracion_dias") or default_days), key=f"dur_{i}")
                with colB:
                    via = st.text_input("Vía", value=p.get("via") or "", key=f"via_{i}")
                    instrucciones = st.text_area("Instrucciones", value=p.get("instrucciones") or "", key=f"inst_{i}")

                p_norm = {"medicamento": med_name, "dosis": dosis_text, "frecuencia": frecuencia, "duracion_dias": duracion}
                events = build_events(p_norm, start_dt=start, default_days=default_days)
                all_events.extend(events)

                activo_manual = st.text_input("Principio activo (para explicación/alternativas)", value=med_name, key=f"active_{i}")
                exp = gpt_text_explain(activo_manual) if activo_manual else {"usos_comunes": [], "precauciones_generales": []}
                usos = exp.get("usos_comunes") or []
                precs = exp.get("precauciones_generales") or []

                selected_symptom = ""
                if usos:
                    selected_symptom = st.selectbox("Síntoma/Enfermedad principal para la tabla", usos, index=0, key=f"sym_{i}")
                else:
                    selected_symptom = st.text_input("Síntoma/Enfermedad (manual)", value="", key=f"symph_{i}")

                dosis_mg_val = parse_mg(dosis_text) or parse_ml(dosis_text)
                qty_label = ""
                warn_text = None
                if dosis_mg_val and not isinstance(dosis_mg_val, bool):
                    same_active = cat[cat["active_ingredient"].str.lower() == safe_lower(activo_manual)]
                    if not same_active.empty:
                        for _, row in same_active.iterrows():
                            qty, unidad, nota = unidades_por_toma(dosis_mg_val, row)
                            if qty and unidad:
                                qty_label = human_qty_label(qty, "ml" if unidad=="ml" else unidad)
                                warn_text = dosage_warning(unidad, qty)
                                break

                if warn_text:
                    st.warning(warn_text)
                    warnings.append(warn_text)

                for e in events:
                    printable_rows.append({
                        "fecha_hora": e["datetime"].replace("T"," "),
                        "medicamento": med_name,
                        "dosis": (qty_label or dosis_text),
                        "sintoma_enfermedad": selected_symptom,
                        "notas": instrucciones
                    })

                with st.expander("Ver usos comunes y precauciones"):
                    st.markdown("**Usos comunes:**")
                    if usos: st.write("- " + "\n- ".join(usos))
                    else: st.write("_No disponibles_")
                    st.markdown("**Precauciones generales:**")
                    if precs: st.write("- " + "\n- ".join(precs))
                    else: st.write("_No disponibles_")

            if all_events:

                st.markdown("## Tabla imprimible de dosis")
                df_print = pd.DataFrame(printable_rows)
                st.dataframe(df_print)

                csv_data = csv_printable(printable_rows)
                st.download_button("Descargar CSV imprimible", data=csv_data, file_name="FarmacIA_agenda_imprimible.csv", mime="text/csv")

# Pestaña 2: Consulta de medicamento (foto o nombre)
with tab2:
    st.subheader("Consulta rápida de medicamento")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Por foto (caja/etiqueta)**")
        up = st.file_uploader("Sube foto", type=["png","jpg","jpeg"], key="box_upload")
        if up is not None:
            st.image(up, caption="Imagen subida", width="stretch")
            data_urls = [encode_image_to_data_url(up.read(), up.type)]
            with st.spinner("Leyendo etiqueta (GPT-4o visión)..."):
                box = gpt_vision_json(data_urls, SYSTEM_BOX, "Extrae la ficha en JSON.")

            if box:
                st.success("Etiqueta extraída")
                st.code(json.dumps(box, ensure_ascii=False, indent=2), language="json")
                ai = box.get("active_ingredient") or ""
                if ai:
                    info = gpt_text_explain(ai)
                    st.markdown("**Usos comunes:**")
                    st.write("- " + "\n- ".join(info.get("usos_comunes", [])) if info.get("usos_comunes") else "_No disponibles_")
                    st.markdown("**Recomendaciones generales:**")
                    st.write("- " + "\n- ".join(info.get("precauciones_generales", [])) if info.get("precauciones_generales") else "_No disponibles_")

    with col2:
        st.markdown("**Por nombre (marca o activo)**")
        name = st.text_input("Escribe el nombre (ej. 'ibuprofeno 400 mg')")
        if st.button("Buscar info"):
            ai_guess = name
            info = gpt_text_explain(ai_guess)
            st.markdown("**Usos comunes:**")
            st.write("- " + "\n- ".join(info.get("usos_comunes", [])) if info.get("usos_comunes") else "_No disponibles_")
            st.markdown("**Recomendaciones generales:**")
            st.write("- " + "\n- ".join(info.get("precauciones_generales", [])) if info.get("precauciones_generales") else "_No disponibles_")

# Pestaña 3: Catálogo y alternativas
with tab3:
    st.subheader("Catálogo local de presentaciones (CSV)")
    st.caption("Puedes subir tu propio CSV con el mismo esquema para ampliar el catálogo.")

    ensure_catalog_template("catalogo_meds.csv")
    df = load_catalog("catalogo_meds.csv")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar plantilla CSV de catálogo",
        data=df.to_csv(index=False),
        file_name="catalogo_meds_template.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### Alternativas (misma sustancia)")
    active_search = st.text_input("Principio activo", value="ibuprofeno")
    dosis_text = st.text_input("Dosis prescrita (ej. '400 mg')", value="400 mg")
    if st.button("Calcular alternativas"):
        dosis_mg = parse_mg(dosis_text)
        if dosis_mg is None:
            st.error("No se pudo interpretar la dosis en mg. Ejemplo válido: '400 mg'")
        else:
            alts = alternativas_misma_sustancia(df, active_search, dosis_mg)
            if not alts:
                st.warning("No hay alternativas en el catálogo para esa sustancia.")
            else:
                warn_rows = []
                for a in alts:
                    qty_num = None
                    m = re.search(r"(\d+(?:\.\d+)?)", a["por_toma"])
                    if m:
                        qty_num = float(m.group(1))
                    w = dosage_warning(a.get("unidad_cruda"), qty_num)
                    warn_rows.append({
                        "presentacion": a["presentation"],
                        "por_toma": a["por_toma"],
                        "nota": a["nota"],
                        "advertencia": w or ""
                    })
                st.dataframe(pd.DataFrame(warn_rows))

st.markdown("""
---
**Aviso:** Esta app no proporciona diagnóstico ni tratamiento. Revisa siempre tus medicamentos con un profesional de la salud.
""")
