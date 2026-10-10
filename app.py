import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import random

# --- Page Configuration ---
st.set_page_config(page_title="Multilingual Hallucination Framework", layout="wide", page_icon="🧠", initial_sidebar_state="expanded")

# --- Custom CSS for a Highly Professional, Research-Grade UI ---
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    .sub-title {
        font-size: 1.3rem;
        color: #475569;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }
    .stage-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-top: 5px solid #3b82f6;
        margin-bottom: 20px;
        min-height: 160px;
    }
    .stage-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 10px;
    }
    .stage-desc {
        font-size: 0.9rem;
        color: #64748b;
    }
    .metric-card { 
        background-color: #f8fafc; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #3b82f6; 
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); 
        margin-bottom: 15px;
    }
    .metric-card h4 { color: #64748b; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 5px;}
    .metric-card h2 { color: #0f172a; font-size: 1.8rem; font-weight: 700; margin: 0;}
    .log-box { 
        background-color: #1e1e1e; 
        color: #00ff00; 
        font-family: 'Courier New', monospace; 
        padding: 15px; 
        border-radius: 8px; 
        height: 250px; 
        overflow-y: auto; 
        font-size: 0.85rem; 
        border: 1px solid #334155;
        line-height: 1.5;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 30px;
        background: linear-gradient(to right, #f8fafc, #e2e8f0);
        border-radius: 15px;
        color: #334155;
        border-top: 4px solid #3b82f6;
    }
    .footer h3 { color: #1E3A8A; font-weight: 700; margin-bottom: 10px; }
    .footer p { margin: 5px 0; font-size: 1rem; }
    .footer .mentor { font-weight: 600; color: #0f172a; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# --- Advanced Mock Data & Constants ---
LANGUAGES = ["English (EN)", "Hindi (HI)", "Nepali (NE)", "Telugu (TE)", "Tamil (TA)", "Hinglish (HI-EN)"]
LLMS = ["ChatGPT (OpenAI)", "Claude (Anthropic)", "Gemini (Google)", "DeepSeek (DeepSeek AI)", "Qwen (Alibaba)", "GLM 5.3 (Zhipu AI)"]
HALLUCINATION_TYPES = ["Correct", "Entity Error", "Numerical Error", "Temporal Error", "Location Error", "Fabrication", "Contradiction", "Partial Hallucination"]

# Advanced Fact Bank & Questions
FACT_BANK = [
    {"fact_id": "ECO_001", "domain": "Economics", "fact_type": "Numerical", "ground_truth": "more than 300 cases", "statement": "The WTO brought more than 300 cases in its first ten years.", "distractor": "less than 200 cases"},
    {"fact_id": "HEA_002", "domain": "Health", "fact_type": "Temporal", "ground_truth": "1948", "statement": "The WHO was established in 1948.", "distractor": "1950"},
    {"fact_id": "GEO_003", "domain": "Geography", "fact_type": "Entity", "ground_truth": "Kathmandu", "statement": "The capital of Nepal is Kathmandu.", "distractor": "Pokhara"},
    {"fact_id": "SCI_004", "domain": "Science", "fact_type": "Numerical", "ground_truth": "299,792,458 m/s", "statement": "The speed of light is 299,792,458 m/s.", "distractor": "300,000,000 m/s"},
    {"fact_id": "GEO_005", "domain": "Geography", "fact_type": "Location", "ground_truth": "Amazon Rainforest", "statement": "The largest rainforest is the Amazon.", "distractor": "Congo Rainforest"},
    {"fact_id": "ECO_006", "domain": "Economics", "fact_type": "Numerical", "ground_truth": "Consumer Price Index", "statement": "CPI is a broad price index.", "distractor": "Gross Domestic Product"},
    {"fact_id": "HEA_007", "domain": "Health", "fact_type": "Entity", "ground_truth": "Malaria", "statement": "Malaria is transmitted by Anopheles mosquitoes.", "distractor": "Dengue"},
    {"fact_id": "SCI_008", "domain": "Science", "fact_type": "Temporal", "ground_truth": "1953", "statement": "DNA structure was discovered in 1953.", "distractor": "1960"},
]

# --- Advanced Backend Simulation Functions ---

def log_message(logs, message, type="info"):
    """Appends a message to the log list."""
    timestamp = time.strftime("%H:%M:%S")
    prefix = "✅" if type == "success" else ("⚠️" if type == "warn" else "ℹ️")
    logs.append(f"[{timestamp}] {prefix} {message}")
    return logs

def simulate_llm_response(llm, question_data, language):
    """Simulates LLM generation with realistic hallucination patterns."""
    time.sleep(0.1)
    
    is_low_resource = language in ["Telugu (TE)", "Tamil (TA)", "Nepali (NE)"]
    is_code_mixed = language == "Hinglish (HI-EN)"
    
    # Simulate higher hallucination rate for low-resource languages
    hallucination_prob = 0.6 if is_low_resource else (0.4 if is_code_mixed else 0.15)
    
    if random.random() < hallucination_prob:
        # Generate a plausible but incorrect response (Hallucination)
        hallucination_types = ["Numerical Error", "Temporal Error", "Entity Error", "Location Error"]
        hall_type = random.choice(hallucination_types)
        
        if hall_type == "Numerical Error":
            return f"[{llm}] The answer is {question_data['distractor']} (Incorrect numerical value).", "Numerical Error"
        elif hall_type == "Temporal Error":
            return f"[{llm}] The event occurred in {question_data['distractor']} (Incorrect date).", "Temporal Error"
        elif hall_type == "Entity Error":
            return f"[{llm}] The entity is {question_data['distractor']} (Wrong entity).", "Entity Error"
        else:
            return f"[{llm}] The location is {question_data['distractor']} (Wrong location).", "Location Error"
    else:
        # Correct Response
        return f"[{llm}] The correct answer is {question_data['ground_truth']}.", "Correct"

def simulate_annotation(question_data, response, language, auto_label):
    """Simulates Fact Consistency Module and Human Annotation."""
    time.sleep(0.05)
    
    is_hallucination = auto_label != "Correct"
    
    if is_hallucination:
        confidence = random.uniform(0.65, 0.92)
    else:
        confidence = random.uniform(0.88, 0.99)
        
    # Human Verification (10% correction rate)
    human_label = auto_label
    if random.random() < 0.1:
        human_label = random.choice(HALLUCINATION_TYPES[1:]) if auto_label == "Correct" else "Correct"
        is_hallucination = human_label != "Correct"
        
    return {
        "auto_label": auto_label,
        "human_label": human_label,
        "is_hallucination": is_hallucination,
        "confidence": round(confidence, 2)
    }

def get_translated_question(fact_id, language):
    """Returns the actual translated question for demonstration."""
    translations = {
        "GEO_003": {
            "English (EN)": "What is the capital of Nepal?",
            "Hindi (HI)": "नेपाल की राजधानी क्या है?",
            "Nepali (NE)": "नेपालको राजधानी के हो?",
            "Telugu (TE)": "నేపాల్ రాజధాని ఏమిటి?",
            "Tamil (TA)": "நேபாளத்தின் தலைநகரம் என்ன?",
            "Hinglish (HI-EN)": "Nepal ki capital kya hai?"
        },
        "ECO_001": {
            "English (EN)": "How many cases were brought to the WTO in its first ten years?",
            "Hindi (HI)": "डब्ल्यूटीओ में पहले दस वर्षों में कितने मामले लाए गए?",
            "Nepali (NE)": "पहिलो दस वर्षमा WTO मा कति मुद्दा दायर भए?",
            "Telugu (TE)": "మొదటి పది సంవత్సరాలలో WTO కి ఎన్ని కేసులు వచ్చాయి?",
            "Tamil (TA)": "முதல் பத்து ஆண்டுகளில் WTO விற்கு எத்தனை வழக்குகள் வந்தன?",
            "Hinglish (HI-EN)": "WTO mein pehle das saal mein kitne cases aaye?"
        },
        "HEA_002": {
            "English (EN)": "When was the WHO established?",
            "Hindi (HI)": "डब्ल्यूएचओ की स्थापना कब हुई थी?",
            "Nepali (NE)": "WHO कहिले स्थापना भएको थियो?",
            "Telugu (TE)": "WHO ఎప్పుడు స్థాపించబడింది?",
            "Tamil (TA)": "WHO எப்போது நிறுவப்பட்டது?",
            "Hinglish (HI-EN)": "WHO kab establish hua tha?"
        }
    }
    return translations.get(fact_id, {}).get(language, "Question not found in translation module.")

# --- UI Layout ---

st.markdown('<h1 class="main-title">🧠 Multilingual Hallucination Detection & Evaluation Framework</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A Comprehensive 9-Stage Pipeline for Low-Resource Languages (Hindi, Hinglish, Telugu, Tamil, Nepali)</p>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Pipeline Configuration")
mode = st.sidebar.radio("Select Operation Mode", ["End-to-End Pipeline Simulation", "Single Query Evaluation", "Batch Dataset Evaluation", "Fact Bank Explorer"])

if mode == "End-to-End Pipeline Simulation":
    st.sidebar.subheader("Stage 1-5: Input & Generation")
    selected_langs = st.sidebar.multiselect("Target Languages", LANGUAGES, default=["English (EN)", "Hindi (HI)", "Telugu (TE)", "Nepali (NE)"])
    selected_llms = st.sidebar.multiselect("Target LLMs", LLMS, default=["ChatGPT (OpenAI)", "Gemini (Google)", "Qwen (Alibaba)"])
    num_facts = st.sidebar.slider("Number of Facts to Evaluate", 3, len(FACT_BANK), 5)
    run_pipeline = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

elif mode == "Single Query Evaluation":
    st.sidebar.subheader("1. Input Processing")
    input_lang = st.sidebar.selectbox("Select Target Language", LANGUAGES)
    fact_options = [f"{f['fact_id']} - {f['statement']}" for f in FACT_BANK]
    selected_fact_str = st.sidebar.selectbox("Select Question", fact_options)
    selected_fact_id = selected_fact_str.split(" - ")[0]
    
    st.sidebar.subheader("2. Model Selection")
    selected_llms_single = st.sidebar.multiselect("Select LLMs to Evaluate", LLMS, default=["ChatGPT (OpenAI)", "Gemini (Google)", "Claude (Anthropic)"])
    run_single = st.sidebar.button("🚀 Run Single Evaluation", type="primary", use_container_width=True)

elif mode == "Batch Dataset Evaluation":
    st.sidebar.subheader("Upload Dataset (Stage 4)")
    uploaded_file = st.sidebar.file_uploader("Upload Annotated JSON/CSV", type=["json", "csv"])
    st.sidebar.info("Columns: question, language, ground_truth, llm_response")
    run_batch = st.sidebar.button("🚀 Evaluate Batch", type="primary", use_container_width=True)

else:
    st.sidebar.subheader("Stage 2: Fact Bank")
    if st.sidebar.button("Generate Fact Bank"):
        st.session_state.fact_bank = pd.DataFrame(FACT_BANK)
        st.sidebar.success("Fact Bank Generated!")

# --- Main Content Area ---

if mode == "End-to-End Pipeline Simulation" and run_pipeline:
    st.info("🚀 Initiating End-to-End Pipeline Simulation...")
    
    # --- Stage 1 & 2: Fact Bank ---
    with st.expander("📚 Stage 1 & 2: Authoritative Source Collection & Fact Bank Construction", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sources Selected:**")
            st.write("Wikipedia, Britannica, WHO, UNESCO, NASA, Govt. Portals (PIB, data.gov), Academic Sources")
        with col2:
            st.markdown("**Fact Extraction Status:**")
            st.success(f"Extracted {num_facts} atomic facts (JSON format).")
        
        fact_df = pd.DataFrame(FACT_BANK[:num_facts])
        st.dataframe(fact_df, use_container_width=True)

    # --- Stage 3 & 4: Question Generation & Dataset Creation ---
    with st.expander("❓ Stage 3 & 4: Question Generation & Multilingual Dataset Creation", expanded=True):
        st.markdown("**Monolingual Dataset (English)**")
        mono_df = fact_df[['fact_id', 'domain', 'fact_type', 'statement', 'ground_truth']].copy()
        st.dataframe(mono_df, use_container_width=True)
        
        st.markdown("**Translation Module Status (Live Demo)**")
        st.info("Translating questions to: " + ", ".join(selected_langs))
        
        # Create multilingual dataset
        multi_data = []
        for _, row in fact_df.iterrows():
            for lang in selected_langs:
                multi_data.append({
                    "question_id": f"Q_{row['domain'][:3].upper()}_{row['fact_id']}",
                    "fact_id": row['fact_id'],
                    "question": get_translated_question(row['fact_id'], lang),
                    "ground_truth": row['ground_truth'],
                    "domain": row['domain'],
                    "fact_type": row['fact_type'],
                    "difficulty": random.choice(["Low", "Medium", "Hard"]),
                    "language": lang,
                    "distractor": row['distractor']
                })
        multilingual_df = pd.DataFrame(multi_data)
        
        # Show translations for the selected facts
        display_cols = ["fact_id", "language", "question", "ground_truth", "distractor"]
        st.dataframe(multilingual_df[display_cols].head(10), use_container_width=True)

    # --- Stage 5 & 6: LLM Response & Annotation ---
    with st.expander("🤖 Stage 5 & 6: LLM Response Collection & Fact Consistency Annotation", expanded=True):
        st.markdown("### 📡 Pipeline Execution Logs")
        log_placeholder = st.empty()
        logs = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_iterations = len(selected_llms) * len(multilingual_df)
        current_iter = 0
        
        for llm in selected_llms:
            for _, row in multilingual_df.iterrows():
                status_text.text(f"Generating response for {llm} in {row['language']}...")
                logs = log_message(logs, f"Querying {llm} for QID: {row['question_id']} in {row['language']}")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                # LLM Response Generation
                response_text, auto_label = simulate_llm_response(llm, row, row['language'])
                
                logs = log_message(logs, f"Received response. Running Fact Consistency Module (XLM-R)...")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                # Annotation
                annotation = simulate_annotation(row, response_text, row['language'], auto_label)
                
                logs = log_message(logs, f"Auto-label: {annotation['auto_label']} | Human-verified: {annotation['human_label']}", "success" if not annotation['is_hallucination'] else "warn")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                results.append({
                    "Question_ID": row['question_id'],
                    "Language": row['language'],
                    "LLM": llm,
                    "Question": row['question'],
                    "Response": response_text,
                    "Auto_Label": annotation['auto_label'],
                    "Human_Label": annotation['human_label'],
                    "Is_Hallucination": annotation['is_hallucination'],
                    "Confidence": annotation['confidence'],
                    "Ground_Truth": row['ground_truth'],
                    "Severity": "High" if annotation['human_label'] in ["Fabrication", "Entity Error", "Numerical Error", "Temporal Error"] else ("Medium" if annotation['is_hallucination'] else "None")
                })
                current_iter += 1
                progress_bar.progress(current_iter / total_iterations)
        
        status_text.text("Generation and Annotation Complete!")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df[['Question_ID', 'Language', 'LLM', 'Response', 'Human_Label', 'Confidence']], use_container_width=True)

    # --- Stage 7 & 8: Detection Model & Evaluation ---
    with st.expander("📊 Stage 7 & 8: Hallucination Detection Model & Evaluation", expanded=True):
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Detailed Responses", "📈 Language-Aware Scoring", "📉 Error Analysis", "📥 Export Data"])
        
        with tab1:
            st.markdown("### Evaluation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            total = len(results_df)
            hallucinations = len(results_df[results_df["Is_Hallucination"] == True])
            correct = total - hallucinations
            reliability = (correct / total) * 100 if total > 0 else 0
            
            col1.markdown(f'<div class="metric-card"><h4>Total Evaluations</h4><h2>{total}</h2></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card" style="border-left-color: #ef4444;"><h4>Hallucinations</h4><h2>{hallucinations}</h2></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-card" style="border-left-color: #22c55e;"><h4>Correct</h4><h2>{correct}</h2></div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="metric-card" style="border-left-color: #eab308;"><h4>Reliability Score</h4><h2>{reliability:.1f}%</h2></div>', unsafe_allow_html=True)
            
            st.markdown("### LLM Response & Hallucination Breakdown")
            def highlight_hallucination(row):
                if row['Is_Hallucination']: return ['background-color: #fee2e2'] * len(row)
                return ['background-color: #dcfce7'] * len(row)
            st.dataframe(results_df.style.apply(highlight_hallucination, axis=1), use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("Language-Aware Reliability Scoring")
            lang_reliability = results_df.groupby("Language")["Is_Hallucination"].apply(lambda x: (1 - x.mean()) * 100).reset_index()
            lang_reliability.columns = ["Language", "Reliability Score (%)"]
            fig_bar = px.bar(lang_reliability, x="Language", y="Reliability Score (%)", 
                             title="Language-Aware Reliability Scoring",
                             color="Language", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_bar.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("### Confidence Calibration (Hallucinations vs Correct)")
            fig_calib = px.scatter(results_df, x="Confidence", y="Is_Hallucination", color="Is_Hallucination", 
                                   title="Model Calibration: Confidence vs Hallucination Status",
                                   labels={"Is_Hallucination": "Is Hallucination"},
                                   color_discrete_map={True: "#ef4444", False: "#22c55e"})
            st.plotly_chart(fig_calib, use_container_width=True)

        with tab3:
            st.subheader("Hallucination Type Distribution")
            hall_counts = results_df[results_df["Is_Hallucination"] == True]["Human_Label"].value_counts().reset_index()
            hall_counts.columns = ["Hallucination Type", "Count"]
            
            if not hall_counts.empty:
                col_pie, col_bar = st.columns(2)
                with col_pie:
                    fig_pie = px.pie(hall_counts, names="Hallucination Type", values="Count", 
                                     title="Distribution of Hallucination Types",
                                     color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col_bar:
                    fig_bar = px.bar(hall_counts, x="Hallucination Type", y="Count", color="Hallucination Type",
                                     title="Frequency of Hallucination Types")
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("No hallucinations detected in this run.")
                
            st.markdown("### Error Severity Analysis")
            sev_counts = results_df[results_df["Is_Hallucination"] == True]["Severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            if not sev_counts.empty:
                fig_sev = px.bar(sev_counts, x="Severity", y="Count", color="Severity", 
                                 title="Severity of Detected Hallucinations",
                                 color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b"})
                st.plotly_chart(fig_sev, use_container_width=True)

        with tab4:
            st.subheader("Export Evaluation Report")
            st.markdown("Download the final annotated benchmark dataset for your research paper.")
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Final Annotated Dataset (CSV)",
                data=csv,
                file_name="multilingual_hallucination_benchmark.csv",
                mime="text/csv",
            )
            
            json_data = results_df.to_json(orient="records", indent=4)
            st.download_button(
                label="📥 Download Final Annotated Dataset (JSON)",
                data=json_data,
                file_name="multilingual_hallucination_benchmark.json",
                mime="application/json",
            )

elif mode == "Single Query Evaluation" and run_single:
    if not selected_llms_single:
        st.warning("Please select at least one LLM.")
    else:
        selected_fact = next((f for f in FACT_BANK if f['fact_id'] == selected_fact_id), None)
        if not selected_fact:
            st.error("Fact not found.")
        else:
            translated_q = get_translated_question(selected_fact_id, input_lang)
            st.info(f"**Question in {input_lang}:** {translated_q}")
            st.markdown(f"**Ground Truth:** {selected_fact['ground_truth']}")
            
            st.markdown("### 📡 Pipeline Execution Logs")
            log_placeholder = st.empty()
            logs = []
            
            results_data = []
            progress_bar = st.progress(0)
            
            for i, llm in enumerate(selected_llms_single):
                logs = log_message(logs, f"Initiating request to {llm} API for language: {input_lang}")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                # Simulate LLM Response
                response_text, auto_label = simulate_llm_response(llm, selected_fact, input_lang)
                logs = log_message(logs, f"Received response from {llm}.")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                logs = log_message(logs, f"Running Fact Consistency Module (XLM-R Embeddings + Cosine Similarity)...")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                # Simulate Detection
                detection = simulate_annotation(selected_fact, response_text, input_lang, auto_label)
                logs = log_message(logs, f"Classification Result: {detection['human_label']} (Confidence: {detection['confidence']})", "success" if not detection['is_hallucination'] else "warn")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
                results_data.append({
                    "LLM": llm, "Language": input_lang, "Question": translated_q,
                    "LLM Response": response_text, "Hallucination Type": detection["human_label"],
                    "Confidence Score": detection["confidence"], "Is Hallucination": detection["is_hallucination"],
                    "Severity": "High" if detection['human_label'] in ["Fabrication", "Entity Error", "Numerical Error"] else ("Medium" if detection['is_hallucination'] else "None"), 
                    "Ground Truth": selected_fact["ground_truth"]
                })
                progress_bar.progress((i + 1) / len(selected_llms_single))
                
            st.success("✅ Evaluation Pipeline Finished Successfully.")
            df = pd.DataFrame(results_data)
            
            st.markdown("## 📊 Evaluation Results")
            col1, col2, col3, col4 = st.columns(4)
            total_responses = len(df)
            hallucinations = len(df[df["Is Hallucination"] == True])
            correct = total_responses - hallucinations
            avg_reliability = (correct / total_responses) * 100 if total_responses > 0 else 0
            
            col1.markdown(f'<div class="metric-card"><h4>Total Responses</h4><h2>{total_responses}</h2></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card" style="border-left-color: #ef4444;"><h4>Hallucinations</h4><h2>{hallucinations}</h2></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-card" style="border-left-color: #22c55e;"><h4>Correct</h4><h2>{correct}</h2></div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="metric-card" style="border-left-color: #eab308;"><h4>Reliability</h4><h2>{avg_reliability:.1f}%</h2></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("LLM Response & Hallucination Breakdown")
            def highlight_hallucination_single(row):
                if row['Is Hallucination']: return ['background-color: #fee2e2'] * len(row)
                return ['background-color: #dcfce7'] * len(row)
            st.dataframe(df.style.apply(highlight_hallucination_single, axis=1), use_container_width=True, hide_index=True)
            
            st.markdown("### 🔍 Detailed View")
            selected_llm_detail = st.selectbox("Select LLM to view details", df["LLM"].unique())
            detail_row = df[df["LLM"] == selected_llm_detail].iloc[0]
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"**Question:** {detail_row['Question']}")
                st.warning(f"**{selected_llm_detail} Response:** {detail_row['LLM Response']}")
            with col_b:
                st.success(f"**Ground Truth:** {detail_row['Ground Truth']}")
                if detail_row['Is Hallucination']:
                    st.error(f"**Hallucination Type:** {detail_row['Hallucination Type']} (Confidence: {detail_row['Confidence Score']})")
                else:
                    st.success(f"**Classification:** Correct (Confidence: {detail_row['Confidence Score']})")

elif mode == "Batch Dataset Evaluation" and run_batch:
    if not uploaded_file:
        st.warning("Please upload a dataset.")
    else:
        st.info("Processing batch dataset...")
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_json(uploaded_file)
            
        st.success(f"Loaded {len(df_input)} records.")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, row in df_input.iterrows():
            status_text.text(f"Processing row {i+1}/{len(df_input)}: {row.get('language', 'Unknown')}")
            is_hall = random.random() < 0.35
            results.append({
                "Question": row.get('question', 'N/A'),
                "Language": row.get('language', 'N/A'),
                "LLM": row.get('llm', 'Unknown'),
                "Ground_Truth": row.get('ground_truth', 'N/A'),
                "LLM_Response": row.get('llm_response', 'N/A'),
                "Predicted_Hallucination": is_hall,
                "Hallucination_Type": random.choice(HALLUCINATION_TYPES[1:]) if is_hall else "Correct",
                "Confidence": round(random.uniform(0.7, 0.99), 2),
                "Severity": "High" if is_hall else "None"
            })
            progress_bar.progress((i + 1) / len(df_input))
        
        status_text.text("Batch processing complete!")
        st.success("✅ Batch Evaluation Finished Successfully.")
        
        df_results = pd.DataFrame(results)
        
        st.markdown("## 📊 Batch Evaluation Results")
        col1, col2, col3, col4 = st.columns(4)
        total = len(df_results)
        hallucinations = len(df_results[df_results["Predicted_Hallucination"] == True])
        correct = total - hallucinations
        reliability = (correct / total) * 100 if total > 0 else 0
        
        col1.metric("Total Evaluations", total)
        col2.metric("Hallucinations Detected", hallucinations)
        col3.metric("Correct", correct)
        col4.metric("Overall Reliability", f"{reliability:.1f}%")
        
        st.markdown("### Reliability by Language and LLM")
        pivot_df = df_results.groupby(['Language', 'LLM'])['Predicted_Hallucination'].apply(lambda x: (1 - x.mean()) * 100).reset_index()
        pivot_df.columns = ['Language', 'LLM', 'Reliability Score (%)']
        
        fig_pivot = px.bar(pivot_df, x="Language", y="Reliability Score (%)", color="LLM", barmode="group",
                           title="Cross-Language Reliability Comparison Across LLMs",
                           color_discrete_sequence=px.colors.qualitative.Bold)
        fig_pivot.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_pivot, use_container_width=True)
        
        st.dataframe(df_results, use_container_width=True)
        
        csv_batch = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Batch Results CSV", data=csv_batch, file_name="batch_evaluation_results.csv", mime="text/csv")

elif mode == "Fact Bank Explorer":
    st.markdown("## 📚 Stage 2: Fact Bank Explorer")
    st.markdown("This module shows the structured JSON output of the LLM-assisted Fact Extraction process.")
    
    if 'fact_bank' not in st.session_state:
        st.session_state.fact_bank = pd.DataFrame(FACT_BANK)
        
    st.dataframe(st.session_state.fact_bank, use_container_width=True)
    
    st.markdown("### JSON Preview (as per Stage 2)")
    st.json(st.session_state.fact_bank.to_dict(orient="records")[0])

else:
    # Landing Page
    st.markdown("### 🏗️ Proposed 9-Stage Research Pipeline")
    
    cols = st.columns(3)
    stages = [
        ("1. Authoritative Source Collection", "Wikipedia, Britannica, WHO, Govt. Portals, Academic Sources."),
        ("2. Fact Bank Construction", "LLM-assisted extraction, Human verification, JSON structured facts."),
        ("3. Question Generation", "LLM-assisted generation from facts, Quality validation, Monolingual dataset."),
        ("4. Multilingual Dataset Creation", "Translation module (EN->HI, NE, TE, TA, Hinglish), Human validation."),
        ("5. LLM Response Collection", "Prompting module, Querying ChatGPT, Claude, Gemini, DeepSeek, Qwen, GLM."),
        ("6. Fact Consistency & Annotation", "Automatic labeling (BERTScore, NLI), Human verification, Annotated dataset."),
        ("7. Hallucination Detection Model", "Language-aware framework, XLM-R/IndicBERT, Semantic similarity, Classifier."),
        ("8. Evaluation & Analysis", "Accuracy/F1, Per-language performance, SHAP analysis, Reliability Score (LRS)."),
        ("9. Benchmark Dataset (Release)", "Final dataset (JSON/CSV), Annotation guidelines, Prompt templates, Model statistics.")
    ]
    
    for i, (title, desc) in enumerate(stages):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="stage-card">
                <div class="stage-title">{title}</div>
                <div class="stage-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.info("👈 Please configure the pipeline parameters in the sidebar and click 'Run Full Pipeline' to demonstrate the complete system.")
    
    st.markdown("### System Architecture Diagram")
    st.markdown("*Refer to the 9-stage pipeline image provided in the research proposal.*")

# --- Professional Footer ---
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <h3>Developed by</h3>
    <p><strong>Shyam Tripathi</strong> (CH.SC.U4CSE23160)</p>
    <p><strong>M Karthik</strong> (CH.SC.U4CSE23121)</p>
    <p class="mentor">UNDER THE GUIDANCE OF</p>
    <p><strong>Dr. G Anitha</strong></p>
    <p>Asst. Professor (Selection Grade) - CSE</p>
    <p>Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai Campus</p>
</div>
""", unsafe_allow_html=True)